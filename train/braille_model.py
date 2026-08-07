"""
BrailleNet — stroke-native braille foundation model.

Architecture:
  Stroke Encoder: point sequence → cell-aligned embeddings
  Cell Decoder:   embeddings → b ∈ {-1, 0, +1}^n per cell
  Algebra Head:   (cell_a, cell_b, op) → result (vector or scalar)

The model operates on strokes and cells — no images.
Strokes are the motor programs; cells are the semantic vectors;
the cell vector b is the lingua franca everything passes through.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# STROKE ENCODER
# ═══════════════════════════════════════════════════════════════════

class StrokeEncoder(nn.Module):
    """Encode a variable-length stroke sequence into cell-aligned embeddings.

    Input:  points (B, T, 2)  — (x, y) coordinates, with SEP (-1,-1) and PAD (-2,-2)
            mask   (B, T)     — True for real points
    Output: cell_embeddings (B, max_cells, d_model)
            cell_pred_mask  (B, max_cells)  — which cells are real
    """

    def __init__(self, d_model=128, nhead=4, num_layers=4, max_cells=32,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_cells = max_cells

        # Point-level projection: (x, y) → d_model
        # We also encode whether a point is a SEP or PAD token
        # Input features: x, y, is_sep, is_pad = 4 dims
        self.point_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Cell segmentation: predict which cell each point belongs to
        # Output: (B, T, max_cells) — soft assignment to cells
        self.cell_assignment = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_cells),
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, points, mask):
        B, T, _ = points.shape

        # Build input features: (x, y, is_sep, is_pad)
        is_sep = ((points[:, :, 0] == -1) & (points[:, :, 1] == -1)).float().unsqueeze(-1)
        is_pad = ((points[:, :, 0] == -2) & (points[:, :, 1] == -2)).float().unsqueeze(-1)
        # Clamp coordinates for SEP/PAD tokens to 0
        xy = points.clone()
        xy[~mask] = 0.0
        xy[(is_sep.squeeze(-1) > 0.5)] = 0.0

        features = torch.cat([xy, is_sep, is_pad], dim=-1)  # (B, T, 4)

        # Project to d_model
        h = self.point_proj(features)  # (B, T, d_model)

        # Add positional encoding
        h = h + self.pos_encoding[:, :T, :]

        # Create attention mask (True = ignore)
        attn_mask = ~mask  # (B, T)

        # Transformer
        h = self.transformer(h, src_key_padding_mask=attn_mask)  # (B, T, d_model)
        h = self.norm(h)

        # Cell assignment: soft-assign each point to a cell
        cell_logits = self.cell_assignment(h)  # (B, T, max_cells)
        # Mask padding tokens
        cell_logits = cell_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        cell_weights = F.softmax(cell_logits, dim=1)  # (B, T, max_cells)

        # Pool: weighted sum of point embeddings per cell
        # cell_weights: (B, T, max_cells) → transpose → (B, max_cells, T)
        cell_embeddings = torch.bmm(
            cell_weights.transpose(1, 2),  # (B, max_cells, T)
            h,                              # (B, T, d_model)
        )  # (B, max_cells, d_model)

        # Determine which cells are active (received significant weight)
        cell_mass = cell_weights.sum(dim=1)  # (B, max_cells)
        cell_pred_mask = cell_mass > 0.1  # threshold for "this cell exists"

        return cell_embeddings, cell_pred_mask


# ═══════════════════════════════════════════════════════════════════
# CELL DECODER
# ═══════════════════════════════════════════════════════════════════

class CellDecoder(nn.Module):
    """Decode cell embeddings to signed dot vectors.

    Input:  cell_embeddings (B, max_cells, d_model)
    Output: dot_logits (B, max_cells, n, 3)  — logits for {-1, 0, +1} per dot
    """

    def __init__(self, d_model=128, n=8):
        super().__init__()
        self.n = n
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n * 3),  # 3 classes per dot
        )

    def forward(self, cell_embeddings):
        B, C, _ = cell_embeddings.shape
        logits = self.decoder(cell_embeddings)  # (B, C, n*3)
        return logits.view(B, C, self.n, 3)     # (B, C, n, 3)


# ═══════════════════════════════════════════════════════════════════
# ALGEBRA HEAD
# ═══════════════════════════════════════════════════════════════════

class AlgebraHead(nn.Module):
    """Algebraic reasoning: (cell_a, cell_b, op) → result.

    Two output heads:
      - Vector head: result ∈ {-1, 0, +1}^n (for add, negate, cancel, update)
      - Scalar head: result ∈ ℝ (for inner product)

    This fixes the Task 2 bottleneck from the composition test.
    """

    def __init__(self, n=8, num_ops=5, hidden=256):
        super().__init__()
        self.n = n
        input_dim = n + n + num_ops  # cell_a + cell_b + op_onehot

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Vector output head: n * 3 logits
        self.vector_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n * 3),
        )

        # Scalar output head
        self.scalar_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cell_a, cell_b, op):
        """
        Args:
            cell_a: (B, n)
            cell_b: (B, n)
            op: (B, num_ops) one-hot

        Returns:
            vector_logits: (B, n, 3)
            scalar_pred: (B, 1)
        """
        x = torch.cat([cell_a, cell_b, op], dim=-1)
        h = self.shared(x)
        vector_logits = self.vector_head(h).view(-1, self.n, 3)
        scalar_pred = self.scalar_head(h)
        return vector_logits, scalar_pred


# ═══════════════════════════════════════════════════════════════════
# BRAILLENET — full model
# ═══════════════════════════════════════════════════════════════════

class BrailleNet(nn.Module):
    """Stroke-native braille foundation model.

    Forward modes:
      - 'perceive': strokes → cell vectors
      - 'algebra':  (cell_a, cell_b, op) → result
    """

    def __init__(self, n=8, d_model=128, nhead=4, num_layers=4,
                 max_cells=32, dropout=0.1):
        super().__init__()
        self.n = n
        self.d_model = d_model

        self.stroke_encoder = StrokeEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            max_cells=max_cells, dropout=dropout,
        )
        self.cell_decoder = CellDecoder(d_model=d_model, n=n)
        self.algebra_head = AlgebraHead(n=n, num_ops=5, hidden=d_model * 2)

    def perceive(self, points, mask):
        """Strokes → cell vectors.

        Returns:
            dot_logits: (B, max_cells, n, 3)
            cell_mask: (B, max_cells)
        """
        cell_embeddings, cell_mask = self.stroke_encoder(points, mask)
        dot_logits = self.cell_decoder(cell_embeddings)
        return dot_logits, cell_mask

    def algebra(self, cell_a, cell_b, op):
        """Algebraic operation on cells.

        Returns:
            vector_logits: (B, n, 3)
            scalar_pred: (B, 1)
        """
        return self.algebra_head(cell_a, cell_b, op)

    def forward(self, mode='perceive', **kwargs):
        if mode == 'perceive':
            return self.perceive(kwargs['points'], kwargs['mask'])
        elif mode == 'algebra':
            return self.algebra(kwargs['cell_a'], kwargs['cell_b'], kwargs['op'])
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def perception_loss(dot_logits, target_cells, cell_mask):
    """Cross-entropy loss for cell decoding.

    Args:
        dot_logits: (B, max_cells, n, 3)
        target_cells: (B, max_cells, n) — values in {-1, 0, +1}
        cell_mask: (B, max_cells) — which cells are real

    Returns:
        loss: scalar
        metrics: dict with per-dot and full-cell accuracy
    """
    B, C, n, _ = dot_logits.shape

    # Map target {-1, 0, +1} → class indices {0, 1, 2}
    target_classes = (target_cells + 1).long()  # -1→0, 0→1, +1→2

    # Flatten for cross-entropy
    logits_flat = dot_logits[cell_mask]  # (num_real_cells, n, 3)
    target_flat = target_classes[cell_mask]  # (num_real_cells, n)

    if logits_flat.shape[0] == 0:
        return torch.tensor(0.0, device=dot_logits.device), {
            'dot_acc': 0.0, 'cell_acc': 0.0, 'num_cells': 0,
        }

    # Reshape for F.cross_entropy: (N*n, 3) vs (N*n,)
    loss = F.cross_entropy(
        logits_flat.reshape(-1, 3),
        target_flat.reshape(-1),
    )

    # Metrics
    with torch.no_grad():
        pred = logits_flat.argmax(dim=-1)  # (num_real_cells, n)
        dot_correct = (pred == target_flat).float()
        dot_acc = dot_correct.mean().item()
        cell_acc = dot_correct.all(dim=-1).float().mean().item()

    return loss, {
        'dot_acc': dot_acc,
        'cell_acc': cell_acc,
        'num_cells': logits_flat.shape[0],
    }


def algebra_loss(vector_logits, scalar_pred, target_result, is_scalar):
    """Combined loss for algebra head.

    Args:
        vector_logits: (B, n, 3)
        scalar_pred: (B, 1)
        target_result: (B, n) — target vector
        is_scalar: (B,) — 1.0 if this sample is a scalar result

    Returns:
        loss, metrics
    """
    B, n, _ = vector_logits.shape
    is_vector = (1.0 - is_scalar)

    # Vector loss: cross-entropy on non-scalar samples
    target_classes = (target_result + 1).long()  # {-1,0,+1} → {0,1,2}
    vec_loss = F.cross_entropy(
        vector_logits.reshape(-1, 3),
        target_classes.reshape(-1),
        reduction='none',
    ).view(B, n)
    # Weight by is_vector
    vec_loss = (vec_loss.mean(dim=1) * is_vector).sum() / max(is_vector.sum(), 1)

    # Scalar loss: MSE on scalar samples
    # For scalar, the target is broadcast to all n dims (target_result[i] = ip/n)
    # Just predict the mean
    scalar_target = target_result[:, 0:1]  # take first element (they're all the same)
    scl_loss = F.mse_loss(scalar_pred, scalar_target, reduction='none').squeeze(-1)
    scl_loss = (scl_loss * is_scalar).sum() / max(is_scalar.sum(), 1)

    loss = vec_loss + scl_loss

    with torch.no_grad():
        # Vector accuracy
        pred_vec = vector_logits.argmax(dim=-1)  # (B, n) — class indices
        correct = (pred_vec == target_classes)
        vec_dot_acc = correct[is_vector > 0.5].float().mean().item() if is_vector.sum() > 0 else 0.0
        vec_cell_acc = correct[is_vector > 0.5].all(dim=-1).float().mean().item() if is_vector.sum() > 0 else 0.0

        # Scalar accuracy (within tolerance)
        if is_scalar.sum() > 0:
            scl_err = (scalar_pred.squeeze(-1) - scalar_target.squeeze(-1)).abs()
            scl_acc = (scl_err[is_scalar > 0.5] < 0.5).float().mean().item()
        else:
            scl_acc = 0.0

    return loss, {
        'vec_dot_acc': vec_dot_acc,
        'vec_cell_acc': vec_cell_acc,
        'scalar_acc': scl_acc,
    }


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    n = 8
    model = BrailleNet(n=n, d_model=128, nhead=4, num_layers=4, max_cells=32)
    print(f"BrailleNet: {model.count_params():,} parameters")

    # Test perception
    B = 4
    T = 256
    points = torch.randn(B, T, 2)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 200:] = False

    dot_logits, cell_mask = model.perceive(points, mask)
    print(f"Perception:")
    print(f"  dot_logits: {dot_logits.shape}")
    print(f"  cell_mask: {cell_mask.shape}, active: {cell_mask.sum(dim=1).tolist()}")

    # Test loss
    target = torch.randint(-1, 2, (B, 32, n)).float()
    loss, metrics = perception_loss(dot_logits, target, cell_mask)
    print(f"  loss: {loss.item():.4f}")
    print(f"  metrics: {metrics}")

    # Test algebra
    cell_a = torch.randn(B, n)
    cell_b = torch.randn(B, n)
    op = F.one_hot(torch.randint(0, 5, (B,)), 5).float()
    vec_logits, scl_pred = model.algebra(cell_a, cell_b, op)
    print(f"\nAlgebra:")
    print(f"  vector_logits: {vec_logits.shape}")
    print(f"  scalar_pred: {scl_pred.shape}")

    target_result = torch.randint(-1, 2, (B, n)).float()
    is_scalar = torch.zeros(B)
    is_scalar[0] = 1.0
    loss, metrics = algebra_loss(vec_logits, scl_pred, target_result, is_scalar)
    print(f"  loss: {loss.item():.4f}")
    print(f"  metrics: {metrics}")

    print(f"\nAll smoke tests passed!")
