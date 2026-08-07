"""
BrailleNet inference engine for the drawing app.

Loads the pre-trained stroke encoder + sketch decoder and provides
real-time sketch→image generation from WebSocket stroke data.

Usage from app.py:
    from inference_engine import InferenceEngine
    engine = InferenceEngine("models/braillenet_pretrained.pth",
                             "models/sketch_decoder_best.pth")
    image_bytes = await engine.generate(stroke_buffer)
"""

import asyncio
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (duplicated from train/ for standalone inference)
# ═══════════════════════════════════════════════════════════════════

class StrokeEncoder(nn.Module):
    """BrailleNet stroke encoder — pre-trained on QuickDraw."""

    def __init__(self, d_model=256, nhead=8, num_layers=6, max_points=512):
        super().__init__()
        self.d_model = d_model
        self.max_points = max_points
        self.point_proj = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_points, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, points, mask):
        B, T, _ = points.shape
        is_sep = ((points[:, :, 0] == -1) & (points[:, :, 1] == -1)).float().unsqueeze(-1)
        is_pad = ((points[:, :, 0] == -2) & (points[:, :, 1] == -2)).float().unsqueeze(-1)
        xy = points.clone()
        xy[~mask] = 0.0
        xy[(is_sep.squeeze(-1) > 0.5)] = 0.0
        features = torch.cat([xy, is_sep, is_pad], dim=-1)
        h = self.point_proj(features)
        h = h + self.pos_encoding[:, :T, :]
        h = self.transformer(h, src_key_padding_mask=~mask)
        return self.norm(h)


class SpatialProjection(nn.Module):
    def __init__(self, d_model=256, spatial_size=16, out_channels=256):
        super().__init__()
        self.spatial_size = spatial_size
        self.out_channels = out_channels
        num_tokens = spatial_size * spatial_size
        self.spatial_queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_channels)

    def forward(self, embeddings, mask=None):
        B = embeddings.shape[0]
        queries = self.spatial_queries.expand(B, -1, -1)
        kpm = ~mask if mask is not None else None
        attn_out, _ = self.cross_attn(queries, embeddings, embeddings, key_padding_mask=kpm)
        attn_out = self.norm(attn_out + queries)
        spatial = self.proj(attn_out).permute(0, 2, 1)
        return spatial.view(B, self.out_channels, self.spatial_size, self.spatial_size)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c, affine=True), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.InstanceNorm2d(c, affine=True),
        )
    def forward(self, x):
        return x + self.net(x)


class UpsampleBlock(nn.Module):
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
    def __init__(self, d_model=256, base_channels=256):
        super().__init__()
        self.spatial_proj = SpatialProjection(d_model, 16, base_channels)
        self.res_blocks = nn.Sequential(*[ResBlock(base_channels) for _ in range(4)])
        self.upsample = nn.Sequential(
            UpsampleBlock(base_channels, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            UpsampleBlock(32, 16),
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, 3, 1), nn.Sigmoid(),
        )

    def forward(self, embeddings, mask=None):
        x = self.spatial_proj(embeddings, mask)
        x = self.res_blocks(x)
        x = self.upsample(x)
        return self.to_rgb(x)


# ═══════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Real-time sketch→image generation from stroke data."""

    def __init__(self, encoder_path: str, decoder_path: str, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"InferenceEngine: loading on {self.device}")

        # Load encoder
        enc_ckpt = torch.load(encoder_path, map_location="cpu", weights_only=True)
        config = enc_ckpt.get("config", {})
        d_model = config.get("d_model", 256)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 6)
        max_points = config.get("max_points", 512)

        self.encoder = StrokeEncoder(d_model, nhead, num_layers, max_points)
        self.encoder.load_state_dict(enc_ckpt["model_state_dict"])
        self.encoder.to(self.device).eval()

        # Load decoder
        dec_ckpt = torch.load(decoder_path, map_location="cpu", weights_only=True)
        self.decoder = SketchDecoder(d_model=d_model)
        self.decoder.load_state_dict(dec_ckpt["decoder_state_dict"])
        self.decoder.to(self.device).eval()

        self.max_points = max_points
        self.d_model = d_model
        self._warmup()

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"  Encoder: {enc_params:,} params")
        print(f"  Decoder: {dec_params:,} params")
        print(f"  Total: {(enc_params + dec_params):,} params")

    def _warmup(self):
        """Run a dummy forward pass to warm up CUDA kernels."""
        with torch.no_grad():
            dummy_points = torch.zeros(1, self.max_points, 2, device=self.device)
            dummy_mask = torch.zeros(1, self.max_points, dtype=torch.bool, device=self.device)
            dummy_mask[:, :10] = True
            dummy_points[:, :10] = torch.rand(1, 10, 2)
            emb = self.encoder(dummy_points, dummy_mask)
            self.decoder(emb, dummy_mask)
        print("  Warmup complete")

    def strokes_to_tensor(self, stroke_buffer: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert WebSocket stroke buffer to model input tensors.

        stroke_buffer: list of {x1, y1, x2, y2, w} dicts (normalized 0-1)
        Returns: (points, mask) ready for the encoder
        """
        # Convert stroke segments to point sequence with SEP tokens
        points_list = []
        prev_x2, prev_y2 = None, None

        for s in stroke_buffer:
            x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]

            # If there's a gap from previous endpoint, insert SEP
            if prev_x2 is not None:
                dist = ((x1 - prev_x2)**2 + (y1 - prev_y2)**2)**0.5
                if dist > 0.01:  # New stroke
                    points_list.append([-1.0, -1.0])

            points_list.append([x1, y1])
            points_list.append([x2, y2])
            prev_x2, prev_y2 = x2, y2

        if not points_list:
            points_list = [[0.5, 0.5]]

        # Convert to tensor
        T = min(len(points_list), self.max_points)
        points = np.full((self.max_points, 2), -2.0, dtype=np.float32)
        points[:T] = np.array(points_list[:T], dtype=np.float32)

        mask = np.zeros(self.max_points, dtype=np.bool_)
        mask[:T] = True

        points_t = torch.from_numpy(points).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        return points_t, mask_t

    @torch.inference_mode()
    def generate(self, stroke_buffer: list[dict], output_size: int = 512) -> tuple[bytes, float]:
        """Generate an image from stroke data.

        Args:
            stroke_buffer: list of {x1, y1, x2, y2, w} dicts
            output_size: output image size

        Returns:
            (jpeg_bytes, inference_time_ms)
        """
        start = time.perf_counter()

        points, mask = self.strokes_to_tensor(stroke_buffer)
        embeddings = self.encoder(points, mask)
        image = self.decoder(embeddings, mask)

        # Convert to PIL Image
        img_np = (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        if output_size != 512:
            pil_img = pil_img.resize((output_size, output_size), Image.LANCZOS)

        # Encode to JPEG
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        elapsed_ms = (time.perf_counter() - start) * 1000
        return jpeg_bytes, elapsed_ms

    async def generate_async(self, stroke_buffer: list[dict],
                             output_size: int = 512) -> tuple[bytes, float]:
        """Async wrapper for generate (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, stroke_buffer, output_size
        )
