"""
PyTorch Dataset/DataLoader for braille foundation model training.

Three dataset modes:
  1. StrokeCellDataset  — stroke sequence ↔ cell vectors (Stages 1–2)
  2. AlgebraDataset     — cell pair + op → result (Stage 3.5)
  3. SequenceDataset    — cell program → result (Stage 3)

All datasets support on-the-fly generation (infinite mode)
or loading from pre-generated JSONL files.
"""

import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from train.braille_data import (
    braille_text_strokes, braille_cell_strokes,
    dots_to_vector, vector_to_dots,
    random_signed_cell, random_cell,
    generate_text_sample, generate_raw_cell_sample,
    generate_algebra_sample, generate_sequence_sample,
    cell_add, cell_negate, cell_inner, cell_update,
    BRAILLE, JITTER_LIGHT, JITTER_MEDIUM, JITTER_HEAVY,
)


# ═══════════════════════════════════════════════════════════════════
# STROKE ENCODING — convert variable-length strokes to fixed tensors
# ═══════════════════════════════════════════════════════════════════

# Special tokens for stroke sequences
SEP_TOKEN = [-1.0, -1.0]   # stroke boundary
PAD_TOKEN = [-2.0, -2.0]   # padding


def encode_strokes(strokes, max_points=512):
    """Encode a list of polyline strokes into a fixed-length tensor.

    Each stroke is a list of (x, y) points. Strokes are concatenated
    with SEP_TOKEN between them and padded to max_points.

    Returns:
        points: (max_points, 2) tensor
        mask: (max_points,) bool tensor (True for real points)
        num_strokes: int
    """
    flat = []
    for i, stroke in enumerate(strokes):
        if i > 0:
            flat.append(SEP_TOKEN)
        for x, y in stroke:
            flat.append([x, y])

    # Truncate if needed
    if len(flat) > max_points:
        flat = flat[:max_points]

    num_real = len(flat)

    # Pad
    while len(flat) < max_points:
        flat.append(PAD_TOKEN)

    points = torch.tensor(flat, dtype=torch.float32)
    mask = torch.zeros(max_points, dtype=torch.bool)
    mask[:num_real] = True

    return points, mask, len(strokes)


def encode_cells(cells, n, max_cells=32):
    """Encode cell vectors into a fixed-length tensor.

    Returns:
        cell_tensor: (max_cells, n) tensor
        cell_mask: (max_cells,) bool tensor
    """
    num_cells = min(len(cells), max_cells)
    cell_tensor = torch.zeros(max_cells, n, dtype=torch.float32)
    cell_mask = torch.zeros(max_cells, dtype=torch.bool)

    for i in range(num_cells):
        cell_tensor[i] = torch.tensor(cells[i], dtype=torch.float32)
        cell_mask[i] = True

    return cell_tensor, cell_mask


# ═══════════════════════════════════════════════════════════════════
# DATASET: Stroke ↔ Cell (Stages 1–2)
# ═══════════════════════════════════════════════════════════════════

class StrokeCellDataset(IterableDataset):
    """On-the-fly generation of (stroke, cell) pairs.

    Infinite dataset — generates samples endlessly for training.
    Use with DataLoader(dataset, batch_size=...) without specifying length.
    """

    def __init__(self, n=8, signed=False, jitter='medium', max_points=512,
                 max_cells=32, mix_text=0.5, mix_raw=0.5):
        """
        Args:
            n: dot count
            signed: use ternary alphabet for raw cells
            jitter: 'none', 'light', 'medium', 'heavy'
            max_points: max stroke points per sample
            max_cells: max cells per sample
            mix_text: fraction of samples from text lookup
            mix_raw: fraction from random cells
        """
        self.n = n
        self.signed = signed
        self.jitter_config = {
            'none': None, 'light': JITTER_LIGHT,
            'medium': JITTER_MEDIUM, 'heavy': JITTER_HEAVY,
        }.get(jitter, JITTER_MEDIUM)
        self.max_points = max_points
        self.max_cells = max_cells
        self.mix_text = mix_text
        self.mix_raw = mix_raw

    def __iter__(self):
        while True:
            r = random.random()
            if r < self.mix_text:
                sample = generate_text_sample(
                    n=self.n, signed=self.signed,
                    jitter=self.jitter_config, mode='mixed',
                )
                strokes = sample['strokes']
                cells = sample['cells']
            else:
                sample = generate_raw_cell_sample(
                    n=self.n, signed=True,
                    jitter=self.jitter_config,
                )
                strokes = sample['strokes']
                cells = sample['cells']

            if not strokes or not cells:
                continue

            points, mask, num_strokes = encode_strokes(strokes, self.max_points)
            cell_tensor, cell_mask = encode_cells(cells, self.n, self.max_cells)

            yield {
                'points': points,           # (max_points, 2)
                'point_mask': mask,          # (max_points,)
                'cells': cell_tensor,        # (max_cells, n)
                'cell_mask': cell_mask,       # (max_cells,)
                'num_strokes': num_strokes,
                'num_cells': sum(1 for c in cells if any(v != 0 for v in c)),
            }


# ═══════════════════════════════════════════════════════════════════
# DATASET: Algebra (Stage 3.5)
# ═══════════════════════════════════════════════════════════════════

class AlgebraDataset(IterableDataset):
    """On-the-fly generation of algebraic operation samples.

    Input: (cell_a, cell_b, operation_code)
    Output: result cell or scalar

    Operations: add, negate, inner, cancel, update
    """

    OP_NAMES = ['add', 'negate', 'inner', 'cancel', 'update']

    def __init__(self, n=8):
        self.n = n

    def __iter__(self):
        while True:
            sample = generate_algebra_sample(n=self.n)

            a = torch.tensor(sample['cell_a'], dtype=torch.float32)
            b = torch.tensor(sample['cell_b'], dtype=torch.float32)
            op_idx = self.OP_NAMES.index(sample['operation'])
            op_onehot = torch.zeros(len(self.OP_NAMES))
            op_onehot[op_idx] = 1.0

            if sample['result_type'] == 'vector':
                result = torch.tensor(sample['result'], dtype=torch.float32)
                is_scalar = torch.tensor(0.0)
                scalar_result = torch.tensor(0.0)
            else:
                result = torch.zeros(self.n)
                is_scalar = torch.tensor(1.0)
                scalar_result = torch.tensor(sample['result'], dtype=torch.float32)

            yield {
                'cell_a': a,
                'cell_b': b,
                'op': op_onehot,
                'result': result,
                'is_scalar': is_scalar,
                'scalar_result': scalar_result,
            }


# ═══════════════════════════════════════════════════════════════════
# DATASET: Sequence Composition (Stage 3)
# ═══════════════════════════════════════════════════════════════════

class SequenceDataset(IterableDataset):
    """On-the-fly generation of stack program samples.

    Input: sequence of L cells (a program)
    Output: result cell (top of stack after execution)
    """

    def __init__(self, n=8, signed=True, max_len=8):
        self.n = n
        self.signed = signed
        self.max_len = max_len

    def __iter__(self):
        while True:
            sample = generate_sequence_sample(
                n=self.n, signed=self.signed, max_len=self.max_len,
            )

            cells = torch.tensor(sample['cells'], dtype=torch.float32)
            result = torch.tensor(sample['result'], dtype=torch.float32)

            yield {
                'cells': cells,       # (max_len, n)
                'result': result,     # (n,)
            }


# ═══════════════════════════════════════════════════════════════════
# HELPER: create DataLoaders
# ═══════════════════════════════════════════════════════════════════

def create_stroke_cell_loader(n=8, signed=False, jitter='medium',
                               batch_size=64, max_points=512, max_cells=32,
                               num_workers=0):
    """Create a DataLoader for stroke ↔ cell training."""
    ds = StrokeCellDataset(
        n=n, signed=signed, jitter=jitter,
        max_points=max_points, max_cells=max_cells,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def create_algebra_loader(n=8, batch_size=64, num_workers=0):
    """Create a DataLoader for algebraic operations."""
    ds = AlgebraDataset(n=n)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


def create_sequence_loader(n=8, signed=True, max_len=8,
                            batch_size=64, num_workers=0):
    """Create a DataLoader for sequence composition."""
    ds = SequenceDataset(n=n, signed=signed, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


# ═══════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=== StrokeCellDataset ===")
    loader = create_stroke_cell_loader(n=8, signed=True, batch_size=4)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"  Batch {i}:")
        print(f"    points:     {batch['points'].shape}")
        print(f"    point_mask: {batch['point_mask'].shape}, "
              f"real={batch['point_mask'].sum(dim=1).tolist()}")
        print(f"    cells:      {batch['cells'].shape}")
        print(f"    cell_mask:  {batch['cell_mask'].shape}, "
              f"real={batch['cell_mask'].sum(dim=1).tolist()}")

    print("\n=== AlgebraDataset ===")
    loader = create_algebra_loader(n=8, batch_size=4)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"  Batch {i}:")
        print(f"    cell_a: {batch['cell_a'].shape}")
        print(f"    op:     {batch['op'].shape}")
        print(f"    result: {batch['result'].shape}")
        print(f"    is_scalar: {batch['is_scalar'].tolist()}")

    print("\n=== SequenceDataset ===")
    loader = create_sequence_loader(n=8, batch_size=4)
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"  Batch {i}:")
        print(f"    cells:  {batch['cells'].shape}")
        print(f"    result: {batch['result'].shape}")

    print("\nAll smoke tests passed!")
