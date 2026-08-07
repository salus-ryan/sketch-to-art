"""
Fast Neural Style Transfer — Modal training script.

Train a feedforward style transfer network (Johnson et al. 2016) on a GPU
via Modal. Downloads the resulting model checkpoint so you can run inference
locally on Apple Silicon (MPS) or CPU.

Usage:
    modal run train/modal_train.py --style-image train/styles/starry_night.jpg
"""

import modal

app = modal.App("style-transfer-train")

# ---------------------------------------------------------------------------
# Modal image: PyTorch + torchvision + Pillow on CUDA
# ---------------------------------------------------------------------------
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "torchvision==0.18.1", "Pillow==10.4.0")
)

# ---------------------------------------------------------------------------
# Persistent volume for caching the COCO dataset between runs
# ---------------------------------------------------------------------------
dataset_volume = modal.Volume.from_name("style-transfer-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------
TRANSFORMER_NET_CODE = '''
import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    """Feedforward image transformation network (Johnson et al. 2016)."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 9, 1),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2),
        )
        # Residual blocks
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # Decoder
        self.decoder = nn.Sequential(
            UpsampleBlock(128, 64, 3, 2),
            UpsampleBlock(64, 32, 3, 2),
            nn.Conv2d(32, 3, 9, 1, 4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, scale):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv2d(in_c, out_c, kernel, 1, pad),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.net(x)
'''

# ---------------------------------------------------------------------------
# Training function (runs on Modal GPU)
# ---------------------------------------------------------------------------
@app.function(
    image=train_image,
    gpu="H100",
    timeout=7200,
    volumes={"/data": dataset_volume},
)
def train(
    style_image_bytes: bytes,
    style_name: str = "custom",
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    content_weight: float = 1e5,
    style_weight: float = 1e10,
    image_size: int = 256,
):
    import io
    import os
    import zipfile
    import urllib.request

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    device = torch.device("cuda")

    # --- Download COCO dataset if not cached ---
    coco_dir = "/data/coco/train2017"
    if not os.path.exists(coco_dir) or len(os.listdir(coco_dir)) < 1000:
        print("Downloading COCO train2017 (~18GB)... this is cached for future runs.")
        url = "http://images.cocodataset.org/zips/train2017.zip"
        zip_path = "/data/coco/train2017.zip"
        os.makedirs("/data/coco", exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("/data/coco")
        os.remove(zip_path)
        print(f"Dataset ready: {len(os.listdir(coco_dir))} images")
    else:
        print(f"Using cached COCO dataset: {len(os.listdir(coco_dir))} images")

    # --- Build model ---
    exec(TRANSFORMER_NET_CODE, globals())
    transformer = TransformerNet().to(device)  # noqa: F821

    # --- VGG16 feature extractor for perceptual loss ---
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    # Feature layers for style/content loss
    style_layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3
    content_layers = [15]  # relu3_3

    def extract_features(x, layers):
        features = []
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in layers:
                features.append(x)
        return features

    def gram_matrix(x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        return torch.bmm(x, x.transpose(1, 2)) / (c * h * w)

    # --- Load style image ---
    style_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    style_img = Image.open(io.BytesIO(style_image_bytes)).convert("RGB")
    style_tensor = style_transform(style_img).unsqueeze(0).to(device)

    # Pre-compute style gram matrices
    style_features = extract_features(style_tensor, style_layers)
    style_grams = [gram_matrix(f).detach() for f in style_features]

    # --- Dataset ---
    content_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    class ImageFolder(Dataset):
        def __init__(self, root, transform):
            self.paths = sorted([
                os.path.join(root, f) for f in os.listdir(root)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            try:
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img)
            except Exception:
                return self.transform(Image.new("RGB", (image_size, image_size)))

    dataset = ImageFolder(coco_dir, content_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=True)

    # --- Train ---
    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    print(f"\nTraining style: {style_name}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {image_size}")
    print(f"Content weight: {content_weight}, Style weight: {style_weight}")
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch\n")

    for epoch in range(epochs):
        transformer.train()
        total_loss = 0
        for i, content_batch in enumerate(loader):
            content_batch = content_batch.to(device)

            output = transformer(content_batch)

            # Content loss
            output_content = extract_features(output, content_layers)
            target_content = extract_features(content_batch, content_layers)
            c_loss = 0
            for oc, tc in zip(output_content, target_content):
                c_loss += nn.functional.mse_loss(oc, tc)

            # Style loss
            output_style = extract_features(output, style_layers)
            s_loss = 0
            for os_feat, sg in zip(output_style, style_grams):
                s_loss += nn.functional.mse_loss(
                    gram_matrix(os_feat), sg.expand(batch_size, -1, -1)
                )

            loss = content_weight * c_loss + style_weight * s_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 200 == 0:
                avg = total_loss / (i + 1)
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(loader)}, "
                      f"Loss: {avg:.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} done — Avg loss: {avg_loss:.4f}\n")

    # --- Save model (CPU state dict for portability) ---
    transformer.cpu()
    buf = io.BytesIO()
    torch.save(transformer.state_dict(), buf)
    model_bytes = buf.getvalue()
    print(f"Model size: {len(model_bytes) / 1024 / 1024:.1f} MB")

    return model_bytes


# ---------------------------------------------------------------------------
# Local entrypoint: reads style image, kicks off training, saves model
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    style_image: str = "train/styles/starry_night.jpg",
    style_name: str = "",
    epochs: int = 2,
    batch_size: int = 64,
):
    from pathlib import Path

    style_path = Path(style_image)
    if not style_path.exists():
        print(f"Style image not found: {style_path}")
        print("Place a style image in train/styles/ and try again.")
        return

    if not style_name:
        style_name = style_path.stem

    print(f"Training style transfer model for: {style_name}")
    print(f"Style image: {style_path}")

    style_bytes = style_path.read_bytes()
    model_bytes = train.remote(
        style_image_bytes=style_bytes,
        style_name=style_name,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save model locally
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{style_name}.pth"
    out_path.write_bytes(model_bytes)
    print(f"\nModel saved to: {out_path}")
    print(f"Run locally with: python train/inference.py --model {out_path}")
