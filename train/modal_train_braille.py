"""
Braille-pretrained style transfer — Modal training script.

Stage 1: Pre-train on synthetically generated 8-dot braille images.
         8-dot braille (computer braille) uses a 2×4 dot grid with 256
         possible patterns — 4× richer than standard 6-dot (64 patterns).
         This teaches the network that spatial dot patterns carry meaning,
         and that precision matters.
Stage 2: Fine-tune on COCO with style loss (same as standard training).

The hypothesis: braille pre-training gives the network a head start on
understanding that spatial structure encodes information, leading to
faster convergence and/or better style transfer quality.

Usage:
    modal run train/modal_train_braille.py --style-image train/styles/starry_night.jpg
"""

import modal

app = modal.App("style-transfer-braille")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "torchvision==0.18.1", "Pillow==10.4.0")
)

dataset_volume = modal.Volume.from_name("style-transfer-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# TransformerNet (same architecture as baseline)
# ---------------------------------------------------------------------------
TRANSFORMER_NET_CODE = '''
import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32, 9, 1),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 128, 3, 2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
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
    braille_pretrain_steps: int = 2000,
    braille_dots: int = 6,
    batch_size: int = 64,
    lr: float = 1e-3,
    content_weight: float = 1e5,
    style_weight: float = 1e10,
    image_size: int = 256,
):
    import io
    import math
    import os
    import random
    import zipfile
    import urllib.request

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, models
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image, ImageDraw

    device = torch.device("cuda")

    # =======================================================================
    # BRAILLE DATA GENERATOR
    # =======================================================================
    num_dots = braille_dots  # 6 (classic: 2x3, 64 patterns) or 8 (computer: 2x4, 256 patterns)
    num_patterns = 2 ** num_dots
    BRAILLE_PATTERNS = []
    for i in range(num_patterns):
        dots = [(i >> bit) & 1 for bit in range(num_dots)]
        BRAILLE_PATTERNS.append(dots)

    # Dot grid layout: 2 columns, 3 rows (6-dot) or 4 rows (8-dot)
    num_rows = num_dots // 2
    DOT_GRID = []
    for r in range(num_rows):
        for c in range(2):
            dx = -1 + c * 2   # -1 or 1
            dy = r - (num_rows - 1) / 2  # centered vertically
            DOT_GRID.append((dx, dy))

    print(f"Braille config: {num_dots}-dot, {num_patterns} patterns, {num_rows} rows")

    def render_braille_image(size=256, cells_per_side=4, jitter=True):
        """Generate an image with random braille characters arranged in a grid.
        Simulates hand-drawn dots with position/size jitter."""
        img = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(img)

        cell_w = size // cells_per_side
        cell_h = size // cells_per_side
        dot_r_base = cell_w // 10

        for row in range(cells_per_side):
            for col in range(cells_per_side):
                pattern = random.choice(BRAILLE_PATTERNS)
                cx = col * cell_w + cell_w // 2
                cy = row * cell_h + cell_h // 2
                spacing = cell_w // 5

                for dot_idx, (dx, dy) in enumerate(DOT_GRID):
                    if pattern[dot_idx]:
                        px = cx + dx * spacing
                        py = cy + dy * spacing

                        if jitter:
                            px += random.randint(-2, 2)
                            py += random.randint(-2, 2)
                            r = dot_r_base + random.randint(-1, 2)
                        else:
                            r = dot_r_base

                        # Vary dot darkness to simulate pressure
                        gray = random.randint(0, 60) if jitter else 0
                        color = (gray, gray, gray)
                        draw.ellipse([px-r, py-r, px+r, py+r], fill=color)

        return img

    class BrailleDataset(Dataset):
        """Generates braille images on-the-fly."""
        def __init__(self, size=256, length=10000, transform=None):
            self.size = size
            self.length = length
            self.transform = transform

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            img = render_braille_image(self.size, cells_per_side=random.randint(2, 6))
            if self.transform:
                img = self.transform(img)
            return img

    # =======================================================================
    # BUILD MODEL
    # =======================================================================
    exec(TRANSFORMER_NET_CODE, globals())
    transformer = TransformerNet().to(device)  # noqa: F821

    # VGG16 for perceptual loss
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    style_layers = [3, 8, 15, 22]
    content_layers = [15]

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

    # Load style image
    style_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    style_img = Image.open(io.BytesIO(style_image_bytes)).convert("RGB")
    style_tensor = style_transform(style_img).unsqueeze(0).to(device)
    style_features = extract_features(style_tensor, style_layers)
    style_grams = [gram_matrix(f).detach() for f in style_features]

    content_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    optimizer = optim.Adam(transformer.parameters(), lr=lr)

    # =======================================================================
    # STAGE 1: BRAILLE PRE-TRAINING
    # =======================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Braille pre-training ({braille_pretrain_steps} steps, {num_dots}-dot)")
    print(f"Teaching network that spatial dot patterns carry meaning...")
    print(f"{num_patterns} unique patterns, {num_rows}-row grid")
    print(f"{'='*60}\n")

    braille_ds = BrailleDataset(size=image_size, length=braille_pretrain_steps * batch_size,
                                 transform=content_transform)
    braille_loader = DataLoader(braille_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

    transformer.train()
    braille_total_loss = 0
    for i, braille_batch in enumerate(braille_loader):
        braille_batch = braille_batch.to(device)
        output = transformer(braille_batch)

        # Content loss — preserve braille dot structure
        output_content = extract_features(output, content_layers)
        target_content = extract_features(braille_batch, content_layers)
        c_loss = sum(nn.functional.mse_loss(oc, tc)
                     for oc, tc in zip(output_content, target_content))

        # Style loss — apply art style to braille
        output_style = extract_features(output, style_layers)
        s_loss = sum(nn.functional.mse_loss(gram_matrix(os_feat),
                     sg.expand(batch_size, -1, -1))
                     for os_feat, sg in zip(output_style, style_grams))

        # Higher content weight during braille phase — structure matters more
        loss = (content_weight * 2) * c_loss + style_weight * s_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        braille_total_loss += loss.item()
        if (i + 1) % 100 == 0:
            avg = braille_total_loss / (i + 1)
            print(f"  Braille step {i+1}/{braille_pretrain_steps}, Loss: {avg:.4f}")

        if i + 1 >= braille_pretrain_steps:
            break

    braille_avg = braille_total_loss / min(len(braille_loader), braille_pretrain_steps)
    print(f"\nBraille pre-training done — Avg loss: {braille_avg:.4f}\n")

    # =======================================================================
    # STAGE 2: COCO FINE-TUNING (same as baseline)
    # =======================================================================
    print(f"{'='*60}")
    print(f"STAGE 2: COCO fine-tuning ({epochs} epochs, batch_size={batch_size})")
    print(f"{'='*60}\n")

    coco_dir = "/data/coco/train2017"
    if not os.path.exists(coco_dir) or len(os.listdir(coco_dir)) < 1000:
        print("Downloading COCO train2017 (~18GB)... cached for future runs.")
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

    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch\n")

    for epoch in range(epochs):
        transformer.train()
        total_loss = 0
        for i, content_batch in enumerate(loader):
            content_batch = content_batch.to(device)
            output = transformer(content_batch)

            output_content = extract_features(output, content_layers)
            target_content = extract_features(content_batch, content_layers)
            c_loss = sum(nn.functional.mse_loss(oc, tc)
                         for oc, tc in zip(output_content, target_content))

            output_style = extract_features(output, style_layers)
            s_loss = sum(nn.functional.mse_loss(gram_matrix(os_feat),
                         sg.expand(batch_size, -1, -1))
                         for os_feat, sg in zip(output_style, style_grams))

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

    # Save
    transformer.cpu()
    buf = io.BytesIO()
    torch.save(transformer.state_dict(), buf)
    model_bytes = buf.getvalue()
    print(f"Model size: {len(model_bytes) / 1024 / 1024:.1f} MB")

    return model_bytes


@app.local_entrypoint()
def main(
    style_image: str = "train/styles/starry_night.jpg",
    style_name: str = "",
    epochs: int = 2,
    batch_size: int = 64,
    braille_pretrain_steps: int = 2000,
    braille_dots: int = 6,
):
    from pathlib import Path

    style_path = Path(style_image)
    if not style_path.exists():
        print(f"Style image not found: {style_path}")
        return

    if not style_name:
        style_name = style_path.stem + f"_braille{braille_dots}"

    print(f"Training BRAILLE-pretrained style transfer: {style_name}")
    print(f"Style image: {style_path}")
    print(f"Braille: {braille_dots}-dot, {braille_pretrain_steps} steps")

    style_bytes = style_path.read_bytes()
    model_bytes = train.remote(
        style_image_bytes=style_bytes,
        style_name=style_name,
        epochs=epochs,
        batch_size=batch_size,
        braille_pretrain_steps=braille_pretrain_steps,
        braille_dots=braille_dots,
    )

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{style_name}.pth"
    out_path.write_bytes(model_bytes)
    print(f"\nModel saved to: {out_path}")
    print(f"Compare with baseline: python train/inference.py --model {out_path}")
