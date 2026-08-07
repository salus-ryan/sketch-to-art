"""
BrailleNet pre-training on QuickDraw strokes — Modal script.

Self-supervised pre-training tasks:
  1. Masked Stroke Prediction — mask random points, predict them from context
  2. Category Prediction — predict which category the strokes belong to (aux task)
  3. Stroke Reconstruction — reconstruct full sequence from cell embeddings

This teaches the stroke encoder to build meaningful spatial embeddings
from raw pen input, which transfers to the sketch→image decoder.

Usage:
    # First, generate data:
    python train/quickdraw_data.py --categories 50 --samples-per-cat 10000 --output data/quickdraw

    # Then pre-train:
    modal run train/modal_pretrain_braillenet.py
"""

import modal

app = modal.App("braillenet-pretrain")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "numpy>=1.24.0")
)

data_volume = modal.Volume.from_name("braillenet-data", create_if_missing=True)
model_volume = modal.Volume.from_name("braillenet-models", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100",
    timeout=14400,  # 4 hours
    volumes={"/data": data_volume, "/models": model_volume},
)
def pretrain(
    num_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 3e-4,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    max_points: int = 512,
    max_cells: int = 32,
    mask_ratio: float = 0.25,
    n_dots: int = 8,
):
    import json
    import os
    import random
    import time
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")

    # ===================================================================
    # DATASET
    # ===================================================================

    class QuickDrawStrokeDataset(Dataset):
        """Load pre-processed QuickDraw stroke .npz files."""

        def __init__(self, data_dir: str, max_points: int = 512):
            self.max_points = max_points
            self.files = []
            self.categories = []
            self.cat_to_idx = {}

            data_path = Path(data_dir) / "strokes"
            if not data_path.exists():
                raise FileNotFoundError(f"No strokes dir at {data_path}")

            for cat_dir in sorted(data_path.iterdir()):
                if not cat_dir.is_dir():
                    continue
                cat_name = cat_dir.name.replace("_", " ")
                if cat_name not in self.cat_to_idx:
                    self.cat_to_idx[cat_name] = len(self.cat_to_idx)
                cat_idx = self.cat_to_idx[cat_name]

                for npz_file in sorted(cat_dir.glob("*.npz")):
                    self.files.append(npz_file)
                    self.categories.append(cat_idx)

            self.num_categories = len(self.cat_to_idx)
            print(f"Dataset: {len(self.files)} samples, {self.num_categories} categories")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            points = data["points"].astype(np.float32)

            # Pad or truncate to max_points
            T = len(points)
            if T > self.max_points:
                points = points[:self.max_points]
                T = self.max_points

            # Create mask (True for real points)
            mask = np.ones(self.max_points, dtype=np.bool_)
            if T < self.max_points:
                padded = np.full((self.max_points, 2), -2.0, dtype=np.float32)
                padded[:T] = points
                points = padded
                mask[T:] = False

            return {
                "points": torch.from_numpy(points),
                "mask": torch.from_numpy(mask),
                "category": self.categories[idx],
                "num_points": T,
            }

    # ===================================================================
    # MODEL (Simplified for pre-training — encoder + prediction heads)
    # ===================================================================

    class StrokeEncoder(nn.Module):
        """Encode stroke points → contextualized embeddings."""

        def __init__(self, d_model=256, nhead=8, num_layers=6, max_points=512):
            super().__init__()
            self.d_model = d_model

            # Point projection: (x, y, is_sep, is_pad) → d_model
            self.point_proj = nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

            # Learned positional encoding
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_points, d_model) * 0.02
            )

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
                activation='gelu',
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, points, mask):
            B, T, _ = points.shape

            # Build features
            is_sep = ((points[:, :, 0] == -1) & (points[:, :, 1] == -1)).float().unsqueeze(-1)
            is_pad = ((points[:, :, 0] == -2) & (points[:, :, 1] == -2)).float().unsqueeze(-1)
            xy = points.clone()
            xy[~mask] = 0.0
            xy[(is_sep.squeeze(-1) > 0.5)] = 0.0

            features = torch.cat([xy, is_sep, is_pad], dim=-1)
            h = self.point_proj(features)
            h = h + self.pos_encoding[:, :T, :]

            attn_mask = ~mask
            h = self.transformer(h, src_key_padding_mask=attn_mask)
            h = self.norm(h)
            return h  # (B, T, d_model)

    class BrailleNetPretrain(nn.Module):
        """Pre-training model with multiple heads."""

        def __init__(self, d_model=256, nhead=8, num_layers=6,
                     max_points=512, num_categories=50):
            super().__init__()
            self.encoder = StrokeEncoder(d_model, nhead, num_layers, max_points)

            # Head 1: Masked point prediction (predict x, y of masked points)
            self.point_predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),  # predict (x, y)
            )

            # Head 2: Category prediction (from [CLS]-like global pooling)
            self.category_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, num_categories),
            )

            # Head 3: Stroke segment prediction (predict if next point is SEP)
            self.sep_predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, points, mask):
            h = self.encoder(points, mask)
            point_pred = self.point_predictor(h)
            # Global pool for category
            # Mask out padding before pooling
            h_masked = h * mask.unsqueeze(-1).float()
            lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            global_emb = h_masked.sum(dim=1) / lengths
            cat_pred = self.category_head(global_emb)
            sep_pred = self.sep_predictor(h).squeeze(-1)
            return point_pred, cat_pred, sep_pred, h

    # ===================================================================
    # TRAINING
    # ===================================================================

    print(f"\nConfig: d_model={d_model}, nhead={nhead}, layers={num_layers}")
    print(f"  batch_size={batch_size}, lr={lr}, epochs={num_epochs}")
    print(f"  mask_ratio={mask_ratio}, max_points={max_points}")

    # Load data
    dataset = QuickDrawStrokeDataset("/data", max_points=max_points)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True,
    )

    # Build model
    model = BrailleNetPretrain(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        max_points=max_points, num_categories=dataset.num_categories,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(loader),
    )

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_point_loss = 0
        epoch_cat_loss = 0
        epoch_sep_loss = 0
        epoch_cat_acc = 0
        num_batches = 0

        for batch in loader:
            points = batch["points"].to(device)
            mask = batch["mask"].to(device)
            categories = torch.tensor(batch["category"], device=device)

            B, T, _ = points.shape

            # === Create masked input ===
            # Only mask real, non-SEP points
            real_point_mask = mask & (points[:, :, 0] != -1) & (points[:, :, 1] != -1)
            # Random mask selection
            rand = torch.rand(B, T, device=device)
            masked = real_point_mask & (rand < mask_ratio)

            # Store targets before masking
            target_points = points.clone()

            # Zero out masked points (model must predict them)
            masked_points = points.clone()
            masked_points[masked] = 0.0

            # === Forward ===
            point_pred, cat_pred, sep_pred, _ = model(masked_points, mask)

            # === Losses ===
            # 1. Masked point prediction (L2 on masked positions)
            if masked.sum() > 0:
                point_loss = F.mse_loss(
                    point_pred[masked],
                    target_points[masked],
                )
            else:
                point_loss = torch.tensor(0.0, device=device)

            # 2. Category prediction
            cat_loss = F.cross_entropy(cat_pred, categories)

            # 3. SEP prediction (binary — is this point a separator?)
            is_sep_target = ((points[:, :, 0] == -1) & (points[:, :, 1] == -1)).float()
            # Only compute on real points (not padding)
            sep_loss = F.binary_cross_entropy_with_logits(
                sep_pred[mask], is_sep_target[mask],
            )

            # Combined loss
            loss = point_loss * 10.0 + cat_loss * 1.0 + sep_loss * 2.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_point_loss += point_loss.item()
            epoch_cat_loss += cat_loss.item()
            epoch_sep_loss += sep_loss.item()
            with torch.no_grad():
                epoch_cat_acc += (cat_pred.argmax(1) == categories).float().mean().item()
            num_batches += 1

        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_point = epoch_point_loss / num_batches
        avg_cat = epoch_cat_loss / num_batches
        avg_sep = epoch_sep_loss / num_batches
        avg_cat_acc = epoch_cat_acc / num_batches
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{num_epochs} ({elapsed/60:.1f}min) | "
              f"loss={avg_loss:.4f} point={avg_point:.4f} "
              f"cat={avg_cat:.4f}(acc={avg_cat_acc:.1%}) sep={avg_sep:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = "/models/braillenet_pretrained.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.encoder.state_dict(),
                "loss": best_loss,
                "config": {
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "max_points": max_points,
                    "num_categories": dataset.num_categories,
                },
            }, save_path)
            print(f"  → Saved best encoder: {save_path}")

    # Save final
    total_time = time.time() - start_time
    final_path = "/models/braillenet_pretrained_final.pth"
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.encoder.state_dict(),
        "full_model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "training_time_s": total_time,
        "config": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "max_points": max_points,
            "num_categories": dataset.num_categories,
            "param_count": param_count,
        },
    }, final_path)
    model_volume.commit()
    print(f"\nTraining complete in {total_time/3600:.2f} hours")
    print(f"Final model saved: {final_path}")
    print(f"Best loss: {best_loss:.4f}")

    return {"best_loss": best_loss, "training_time_h": total_time / 3600}


@app.local_entrypoint()
def main():
    result = pretrain.remote()
    print(f"\nResult: {result}")
