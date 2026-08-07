"""
Braille Composition Law Test
=============================

Can a small transformer learn the composition law of signed n-dot braille?

Three tasks of increasing difficulty:

Task 1 — DOT FACTORIZATION
  Input:  a single cell b ∈ {-1,0,+1}^n
  Output: which dots are active, denied, absent (identity mapping)
  This tests whether the model learns factored dot representations
  vs memorizing all 3^n cells.

Task 2 — ALGEBRAIC OPERATIONS
  Input:  two cells (b₁, b₂) and an operation (+, -, inner product)
  Output: the result cell or scalar
  This tests whether the model learns algebraic structure:
    (+d) + (-d) = 0
    negation: -b
    inner product: ⟨b₁, b₂⟩

Task 3 — SEQUENCE COMPOSITION
  Input:  L cells encoding a "program" (e.g., braille arithmetic)
  Output: the result
  This tests whether the model can learn composition over sequences.

We measure:
  - Accuracy on seen patterns vs unseen patterns (generalization)
  - Accuracy vs n (does factorization scale?)
  - Learning curves (how fast does composition emerge?)

Usage:
    python train/braille_composition_test.py [--n 8] [--signed] [--task 1|2|3]
"""

import argparse
import json
import math
import random
import time
from itertools import product as cartesian_product
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Braille Cell Representation ───

def all_cells(n, signed=False):
    """Generate all possible n-dot cells."""
    alphabet = [-1, 0, 1] if signed else [0, 1]
    return list(cartesian_product(alphabet, repeat=n))


def cell_to_tensor(cell):
    """Convert a cell tuple to a float tensor."""
    return torch.tensor(cell, dtype=torch.float32)


# ─── Task 1: Dot Factorization ───

class DotFactorizationDataset(Dataset):
    """
    Input: cell b ∈ {-1,0,+1}^n
    Output: same cell (identity — but the model must learn the
            factored representation, not memorize all 3^n cells)

    The test: train on 70% of cells, evaluate on held-out 30%.
    If the model has learned factorization, it generalizes perfectly.
    If it memorized, it fails on unseen cells.
    """
    def __init__(self, cells):
        self.cells = cells

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        c = cell_to_tensor(self.cells[idx])
        return c, c  # input = output (autoencoder)


# ─── Task 2: Algebraic Operations ───

class AlgebraDataset(Dataset):
    """
    Input: (cell_a, cell_b, op_code)
    Output: result

    Operations:
      0: addition       → clamp(a + b, -1, 1)
      1: negation of a  → -a
      2: inner product   → ⟨a, b⟩ (scalar, normalized to [-1, 1])
      3: cancellation test → a + (-a) should = 0
    """
    def __init__(self, cells, n, num_samples=10000):
        self.n = n
        self.samples = []
        for _ in range(num_samples):
            a = random.choice(cells)
            b = random.choice(cells)
            op = random.randint(0, 3)

            a_t = torch.tensor(a, dtype=torch.float32)
            b_t = torch.tensor(b, dtype=torch.float32)

            if op == 0:  # addition
                result = torch.clamp(a_t + b_t, -1, 1)
            elif op == 1:  # negation
                result = -a_t
            elif op == 2:  # inner product
                ip = (a_t * b_t).sum()
                result = torch.full((n,), ip / n)  # broadcast scalar
            else:  # cancellation: a + (-a)
                b_t = -a_t
                result = torch.zeros(n)

            op_t = F.one_hot(torch.tensor(op), num_classes=4).float()
            inp = torch.cat([a_t, b_t, op_t])
            self.samples.append((inp, result))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─── Task 3: Sequence Composition ───

class SequenceDataset(Dataset):
    """
    Input: sequence of L cells, interpreted as a stack program:
      - Cell with only dot 1 = PUSH next cell
      - Cell with only dot 2 = ADD top two
      - Cell with only dot 3 = NEGATE top
      - Cell with only dots 1,2 = INNER PRODUCT top two
      - Any other cell = literal (pushed to stack)

    Output: top of stack after execution

    This tests whether the model can learn composition over sequences.
    """
    def __init__(self, n, signed=False, num_samples=5000, max_len=8):
        self.n = n
        self.samples = []
        alphabet = [-1, 0, 1] if signed else [0, 1]

        for _ in range(num_samples):
            # Generate a random program and execute it
            seq_len = random.randint(3, max_len)
            cells = []
            stack = []

            for _ in range(seq_len):
                # Random cell
                cell = tuple(random.choice(alphabet) for _ in range(n))
                cells.append(cell)

                # Execute
                c_t = torch.tensor(cell, dtype=torch.float32)
                if cell == tuple([1] + [0] * (n - 1)) and len(stack) >= 0:
                    pass  # PUSH: next cell will be pushed
                elif cell == tuple([0, 1] + [0] * (n - 2)) and len(stack) >= 2:
                    a, b = stack.pop(), stack.pop()
                    stack.append(torch.clamp(a + b, -1, 1))
                elif cell == tuple([0, 0, 1] + [0] * (n - 3)) and len(stack) >= 1:
                    stack.append(-stack.pop())
                elif cell == tuple([1, 1] + [0] * (n - 2)) and len(stack) >= 2:
                    a, b = stack.pop(), stack.pop()
                    ip = (a * b).sum() / n
                    stack.append(torch.full((n,), ip.item()))
                else:
                    stack.append(c_t)

            if not stack:
                stack.append(torch.zeros(n))

            # Pad sequence to max_len
            while len(cells) < max_len:
                cells.append(tuple([0] * n))

            seq_tensor = torch.stack([cell_to_tensor(c) for c in cells[:max_len]])
            result = stack[-1]
            self.samples.append((seq_tensor, result))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─── Models ───

class FactorizationNet(nn.Module):
    """Small MLP for dot factorization (Task 1)."""
    def __init__(self, n, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class AlgebraNet(nn.Module):
    """MLP for algebraic operations (Task 2)."""
    def __init__(self, n, hidden=128):
        super().__init__()
        input_dim = n + n + 4  # cell_a + cell_b + op_onehot
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class CompositionTransformer(nn.Module):
    """Small transformer for sequence composition (Task 3)."""
    def __init__(self, n, max_len=8, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.n = n
        self.input_proj = nn.Linear(n, d_model)
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, n)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x: (batch, seq_len, n)
        h = self.input_proj(x) + self.pos_embed[:x.size(1)]
        h = self.transformer(h)
        h = h[:, -1, :]  # take last position
        return self.tanh(self.output_proj(h))


# ─── Training ───

def train_task(model, train_loader, test_loader, epochs=100, lr=1e-3, task_name=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        train_loss = total_loss / len(train_loader)
        test_loss = 0
        test_acc = 0
        test_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                test_loss += F.mse_loss(pred, y).item()
                # Accuracy: round to nearest {-1, 0, 1} and compare
                pred_rounded = torch.round(pred).clamp(-1, 1)
                correct = (pred_rounded == y).all(dim=-1).sum().item()
                test_acc += correct
                test_count += y.size(0)

        test_loss /= len(test_loader)
        accuracy = test_acc / test_count

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        })

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [{task_name}] epoch {epoch:3d} | "
                  f"train_loss={train_loss:.6f} | "
                  f"test_loss={test_loss:.6f} | "
                  f"test_acc={accuracy:.4f}")

    return results


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Braille Composition Law Test")
    parser.add_argument("--n", type=int, default=8, help="Number of dots")
    parser.add_argument("--signed", action="store_true", help="Use {-1,0,+1} instead of {0,1}")
    parser.add_argument("--task", type=int, default=0, help="Task: 0=all, 1=factorize, 2=algebra, 3=compose")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per task")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--sweep", action="store_true", help="Sweep n from 4 to 16")
    args = parser.parse_args()

    all_results = {}

    n_values = list(range(4, 17, 2)) if args.sweep else [args.n]

    for n in n_values:
        print(f"\n{'='*60}")
        sign_label = "signed {-1,0,+1}" if args.signed else "binary {0,1}"
        total_cells = 3**n if args.signed else 2**n
        print(f"n={n}-dot, {sign_label}, {total_cells} total cells")
        print(f"{'='*60}")

        cells = all_cells(n, args.signed)

        # Limit cells for large n (can't enumerate all 3^16 = 43M)
        if len(cells) > 50000:
            print(f"  Sampling 50000 of {len(cells)} cells")
            cells = random.sample(cells, 50000)

        n_results = {}

        # ── Task 1: Dot Factorization ──
        if args.task in [0, 1]:
            print(f"\n── Task 1: Dot Factorization (n={n}) ──")
            random.shuffle(cells)
            split = int(len(cells) * 0.7)
            train_cells = cells[:split]
            test_cells = cells[split:]
            print(f"  Train: {len(train_cells)} cells, Test: {len(test_cells)} cells")

            train_ds = DotFactorizationDataset(train_cells)
            test_ds = DotFactorizationDataset(test_cells)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=args.batch_size)

            model = FactorizationNet(n, hidden=max(64, n * 8))
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {param_count:,}")

            t0 = time.time()
            results = train_task(model, train_dl, test_dl, args.epochs,
                                task_name=f"Factor-{n}")
            elapsed = time.time() - t0

            final_acc = results[-1]["test_accuracy"]
            print(f"\n  RESULT: {final_acc:.4f} accuracy on UNSEEN cells ({elapsed:.1f}s)")
            print(f"  {'FACTORIZED ✓' if final_acc > 0.95 else 'MEMORIZED ✗'}")
            n_results["task1_factorization"] = {
                "accuracy": final_acc,
                "factorized": final_acc > 0.95,
                "train_cells": len(train_cells),
                "test_cells": len(test_cells),
                "params": param_count,
                "seconds": elapsed,
                "curve": results,
            }

        # ── Task 2: Algebraic Operations ──
        if args.task in [0, 2]:
            print(f"\n── Task 2: Algebraic Operations (n={n}) ──")
            train_ds = AlgebraDataset(cells, n, num_samples=10000)
            test_ds = AlgebraDataset(cells, n, num_samples=2000)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=args.batch_size)

            model = AlgebraNet(n, hidden=max(128, n * 16))
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {param_count:,}")

            t0 = time.time()
            results = train_task(model, train_dl, test_dl, args.epochs,
                                task_name=f"Algebra-{n}")
            elapsed = time.time() - t0

            final_acc = results[-1]["test_accuracy"]
            print(f"\n  RESULT: {final_acc:.4f} accuracy ({elapsed:.1f}s)")
            print(f"  {'ALGEBRAIC ✓' if final_acc > 0.90 else 'PARTIAL ✗'}")
            n_results["task2_algebra"] = {
                "accuracy": final_acc,
                "algebraic": final_acc > 0.90,
                "params": param_count,
                "seconds": elapsed,
                "curve": results,
            }

        # ── Task 3: Sequence Composition ──
        if args.task in [0, 3]:
            print(f"\n── Task 3: Sequence Composition (n={n}) ──")
            train_ds = SequenceDataset(n, args.signed, num_samples=8000, max_len=8)
            test_ds = SequenceDataset(n, args.signed, num_samples=2000, max_len=8)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=args.batch_size)

            model = CompositionTransformer(n, max_len=8, d_model=64, nhead=4, num_layers=3)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Model params: {param_count:,}")

            t0 = time.time()
            results = train_task(model, train_dl, test_dl, args.epochs,
                                lr=3e-4, task_name=f"Compose-{n}")
            elapsed = time.time() - t0

            final_acc = results[-1]["test_accuracy"]
            print(f"\n  RESULT: {final_acc:.4f} accuracy ({elapsed:.1f}s)")
            print(f"  {'COMPOSITIONAL ✓' if final_acc > 0.80 else 'LIMITED ✗'}")
            n_results["task3_composition"] = {
                "accuracy": final_acc,
                "compositional": final_acc > 0.80,
                "params": param_count,
                "seconds": elapsed,
                "curve": results,
            }

        all_results[f"n={n}"] = n_results

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for n_key, n_res in all_results.items():
        print(f"\n{n_key}:")
        for task_key, task_res in n_res.items():
            status = "✓" if task_res.get("factorized") or task_res.get("algebraic") or task_res.get("compositional") else "✗"
            print(f"  {task_key}: {task_res['accuracy']:.4f} {status} ({task_res['params']:,} params, {task_res['seconds']:.1f}s)")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"composition_test_n{args.n}{'_signed' if args.signed else ''}.json"
    # Strip curves for JSON (too large)
    save_results = {}
    for n_key, n_res in all_results.items():
        save_results[n_key] = {}
        for task_key, task_res in n_res.items():
            save_results[n_key][task_key] = {k: v for k, v in task_res.items() if k != "curve"}
    with open(out_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
