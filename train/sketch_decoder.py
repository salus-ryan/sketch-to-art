"""
Sketch→Image Decoder

Takes BrailleNet stroke embeddings and generates a 512×512 image.
Architecture: Embedding projection → spatial upsampling → CNN decoder.

The key idea: BrailleNet embeddings encode the *meaning* of strokes
(structure, composition, category) rather than just pixel locations.
This richer conditioning signal should enable faster learning.

Two modes:
  - 'embedding': Takes pre-computed BrailleNet embeddings
  - 'end2end': Takes raw stroke points, encodes through BrailleNet, then decodes
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialProjection(nn.Module):
    """Project sequence of embeddings into a 2D spatial feature map.

    Takes (B, T, d_model) → (B, channels, H, W) by:
    1. Pool/compress sequence into fixed number of spatial tokens
    2. Reshape into 2D grid
    3. Upsample
    """

    def __init__(self, d_model=256, spatial_size=16, out_channels=256):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        num_tokens = spatial_size * spatial_size  # 256 tokens for 16x16

        # Cross-attention: learned spatial queries attend to stroke embeddings
        self.spatial_queries = nn.Parameter(
            torch.randn(1, num_tokens, d_model) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, batch_first=True, dropout=0.1,
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_channels)

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, T, d_model)
            mask: (B, T) bool — True for valid positions
        """
        B = embeddings.shape[0]
        queries = self.spatial_queries.expand(B, -1, -1)

        # Cross-attention: spatial queries attend to stroke embeddings
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.cross_attn(
            queries, embeddings, embeddings,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.norm(attn_out + queries)

        # Project and reshape to 2D
        spatial = self.proj(attn_out)  # (B, H*W, out_channels)
        spatial = spatial.permute(0, 2, 1)  # (B, C, H*W)
        spatial = spatial.view(B, self.out_channels, self.spatial_size, self.spatial_size)
        return spatial


class ResBlock(nn.Module):
    """Residual block with instance norm."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.net(x)


class UpsampleBlock(nn.Module):
    """Upsample 2× with conv."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SketchDecoder(nn.Module):
    """Decode BrailleNet embeddings → RGB image.

    Pipeline:
        Stroke embeddings (B, T, d_model)
        → SpatialProjection → (B, 256, 16, 16)
        → ResBlocks
        → Upsample to 512×512
        → Conv → RGB

    Total: ~8M parameters (fits comfortably on 4070)
    """

    def __init__(self, d_model=256, base_channels=256, output_size=512):
        super().__init__()
        self.output_size = output_size

        # 16×16 spatial features from embeddings
        self.spatial_proj = SpatialProjection(
            d_model=d_model, spatial_size=16, out_channels=base_channels,
        )

        # Residual refinement at 16×16
        self.res_blocks = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
        )

        # Upsample: 16 → 32 → 64 → 128 → 256 → 512
        self.upsample = nn.Sequential(
            UpsampleBlock(base_channels, 256),   # 32×32
            UpsampleBlock(256, 128),             # 64×64
            UpsampleBlock(128, 64),              # 128×128
            UpsampleBlock(64, 32),               # 256×256
            UpsampleBlock(32, 16),               # 512×512
        )

        # Final conv to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: (B, T, d_model) — BrailleNet stroke embeddings
            mask: (B, T) — valid point mask

        Returns:
            image: (B, 3, 512, 512) — generated RGB image [0, 1]
        """
        spatial = self.spatial_proj(embeddings, mask)  # (B, 256, 16, 16)
        spatial = self.res_blocks(spatial)
        x = self.upsample(spatial)
        return self.to_rgb(x)


class SketchToImage(nn.Module):
    """Full end-to-end model: strokes → image.

    Combines:
    1. BrailleNet stroke encoder (pre-trained, optionally frozen)
    2. SketchDecoder (trained from scratch)
    """

    def __init__(self, encoder, decoder, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, points, mask):
        """
        Args:
            points: (B, T, 2) — stroke points
            mask: (B, T) — valid point mask

        Returns:
            image: (B, 3, 512, 512)
        """
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(points, mask)
        else:
            embeddings = self.encoder(points, mask)

        return self.decoder(embeddings, mask)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ═══════════════════════════════════════════════════════════════════
# LOSSES
# ═══════════════════════════════════════════════════════════════════

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # Use layers up to relu4_3
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23], # relu4_3
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        """Compute perceptual loss between predicted and target images."""
        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    d_model = 256
    B = 2
    T = 256

    # Test spatial projection
    emb = torch.randn(B, T, d_model)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 200:] = False

    proj = SpatialProjection(d_model=d_model, spatial_size=16, out_channels=256)
    spatial = proj(emb, mask)
    print(f"SpatialProjection: {emb.shape} → {spatial.shape}")

    # Test full decoder
    decoder = SketchDecoder(d_model=d_model)
    img = decoder(emb, mask)
    print(f"SketchDecoder: {emb.shape} → {img.shape}")
    print(f"  Parameters: {sum(p.numel() for p in decoder.parameters()):,}")

    # Test perceptual loss
    target = torch.randn(B, 3, 512, 512).clamp(0, 1)
    perc_loss = PerceptualLoss()
    loss = perc_loss(img, target)
    print(f"  Perceptual loss: {loss.item():.4f}")

    print("\nAll smoke tests passed!")
