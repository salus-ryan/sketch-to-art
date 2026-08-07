"""
BrailleNet training script — local or Modal.

Stage 1: Stroke → Cell perception
  Train the stroke encoder + cell decoder on synthetic data.
  Target: >95% full-cell accuracy on unseen n=8 signed braille.

Stage 3.5: Ternary algebra
  Train the algebra head: (cell_a, cell_b, op) → result.
  Target: >90% on all 5 operations.

Usage:
    # Local training (CPU/MPS)
    PYTHONPATH=. python train/train_braillenet.py --stage 1 --epochs 50

    # Both stages
    PYTHONPATH=. python train/train_braillenet.py --stage both --epochs 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train.braille_model import BrailleNet, perception_loss, algebra_loss
from train.braille_dataset import (
    create_stroke_cell_loader,
    create_algebra_loader,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_perception(model, device, n=8, epochs=50, lr=3e-4, batch_size=32,
                     steps_per_epoch=200, jitter='medium', log_every=20,
                     max_points=256):
    """Stage 1: Train stroke → cell perception."""
    print(f"\n{'='*60}")
    print(f"STAGE 1: Stroke → Cell Perception")
    print(f"  n={n}, jitter={jitter}, lr={lr}, batch_size={batch_size}")
    print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs = {steps_per_epoch * epochs} steps")
    print(f"{'='*60}\n")

    model.to(device)
    optimizer = torch.optim.AdamW(
        list(model.stroke_encoder.parameters()) + list(model.cell_decoder.parameters()),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch,
    )

    # Create data loaders — one clean for eval, one jittered for training
    train_loader = create_stroke_cell_loader(
        n=n, signed=True, jitter=jitter, batch_size=batch_size,
        max_points=max_points, max_cells=32,
    )
    eval_loader = create_stroke_cell_loader(
        n=n, signed=True, jitter='none', batch_size=batch_size,
        max_points=max_points, max_cells=32,
    )

    train_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    results = []
    best_cell_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_dot_acc = 0
        epoch_cell_acc = 0
        epoch_cells = 0

        for step in range(steps_per_epoch):
            batch = next(train_iter)
            points = batch['points'].to(device)
            mask = batch['point_mask'].to(device)
            target = batch['cells'].to(device)
            cell_mask = batch['cell_mask'].to(device)

            dot_logits, pred_mask = model.perceive(points, mask)

            # Use ground-truth cell mask for loss (we know which cells are real)
            loss, metrics = perception_loss(dot_logits, target, cell_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_dot_acc += metrics['dot_acc'] * metrics['num_cells']
            epoch_cell_acc += metrics['cell_acc'] * metrics['num_cells']
            epoch_cells += metrics['num_cells']

        # Epoch stats
        avg_loss = epoch_loss / steps_per_epoch
        avg_dot_acc = epoch_dot_acc / max(epoch_cells, 1)
        avg_cell_acc = epoch_cell_acc / max(epoch_cells, 1)

        # Eval on clean data
        model.eval()
        eval_dot_acc = 0
        eval_cell_acc = 0
        eval_cells = 0
        with torch.no_grad():
            for _ in range(50):
                batch = next(eval_iter)
                points = batch['points'].to(device)
                mask = batch['point_mask'].to(device)
                target = batch['cells'].to(device)
                cell_mask = batch['cell_mask'].to(device)

                dot_logits, _ = model.perceive(points, mask)
                _, metrics = perception_loss(dot_logits, target, cell_mask)
                eval_dot_acc += metrics['dot_acc'] * metrics['num_cells']
                eval_cell_acc += metrics['cell_acc'] * metrics['num_cells']
                eval_cells += metrics['num_cells']

        eval_dot = eval_dot_acc / max(eval_cells, 1)
        eval_cell = eval_cell_acc / max(eval_cells, 1)

        if eval_cell > best_cell_acc:
            best_cell_acc = eval_cell

        result = {
            'epoch': epoch,
            'train_loss': round(avg_loss, 6),
            'train_dot_acc': round(avg_dot_acc, 4),
            'train_cell_acc': round(avg_cell_acc, 4),
            'eval_dot_acc': round(eval_dot, 4),
            'eval_cell_acc': round(eval_cell, 4),
            'best_cell_acc': round(best_cell_acc, 4),
            'lr': scheduler.get_last_lr()[0],
        }
        results.append(result)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} | "
                  f"train dot={avg_dot_acc:.4f} cell={avg_cell_acc:.4f} | "
                  f"eval dot={eval_dot:.4f} cell={eval_cell:.4f} | "
                  f"best={best_cell_acc:.4f}")

    return results, best_cell_acc


def train_algebra(model, device, n=8, epochs=50, lr=3e-4, batch_size=64,
                  steps_per_epoch=200, log_every=10):
    """Stage 3.5: Train ternary algebra head."""
    print(f"\n{'='*60}")
    print(f"STAGE 3.5: Ternary State Update Algebra")
    print(f"  n={n}, lr={lr}, batch_size={batch_size}")
    print(f"  Operations: add, negate, inner, cancel, update")
    print(f"{'='*60}\n")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.algebra_head.parameters(),
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch,
    )

    train_loader = create_algebra_loader(n=n, batch_size=batch_size)
    train_iter = iter(train_loader)

    results = []
    best_vec_acc = 0.0
    best_scl_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_vec_dot = 0
        epoch_vec_cell = 0
        epoch_scl = 0
        n_vec = 0
        n_scl = 0

        for step in range(steps_per_epoch):
            batch = next(train_iter)
            cell_a = batch['cell_a'].to(device)
            cell_b = batch['cell_b'].to(device)
            op = batch['op'].to(device)
            target = batch['result'].to(device)
            is_scalar = batch['is_scalar'].to(device)

            vec_logits, scl_pred = model.algebra(cell_a, cell_b, op)
            loss, metrics = algebra_loss(vec_logits, scl_pred, target, is_scalar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            bv = (is_scalar < 0.5).sum().item()
            bs = (is_scalar > 0.5).sum().item()
            epoch_vec_dot += metrics['vec_dot_acc'] * bv
            epoch_vec_cell += metrics['vec_cell_acc'] * bv
            epoch_scl += metrics['scalar_acc'] * bs
            n_vec += bv
            n_scl += bs

        avg_loss = epoch_loss / steps_per_epoch
        avg_vec_dot = epoch_vec_dot / max(n_vec, 1)
        avg_vec_cell = epoch_vec_cell / max(n_vec, 1)
        avg_scl = epoch_scl / max(n_scl, 1)

        best_vec_acc = max(best_vec_acc, avg_vec_cell)
        best_scl_acc = max(best_scl_acc, avg_scl)

        result = {
            'epoch': epoch,
            'loss': round(avg_loss, 6),
            'vec_dot_acc': round(avg_vec_dot, 4),
            'vec_cell_acc': round(avg_vec_cell, 4),
            'scalar_acc': round(avg_scl, 4),
        }
        results.append(result)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} | "
                  f"vec_dot={avg_vec_dot:.4f} vec_cell={avg_vec_cell:.4f} | "
                  f"scalar={avg_scl:.4f}")

    return results, best_vec_acc, best_scl_acc


def main():
    parser = argparse.ArgumentParser(description='Train BrailleNet')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--stage', default='both', choices=['1', '3.5', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--jitter', default='medium')
    parser.add_argument('--steps-per-epoch', type=int, default=200)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--cpu', action='store_true', help='Force CPU (avoids MPS fallback)')
    parser.add_argument('--max-points', type=int, default=256, help='Max stroke points per sample')
    parser.add_argument('--log-every', type=int, default=5, help='Log every N epochs')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else get_device()
    print(f"Device: {device}")

    model = BrailleNet(
        n=args.n, d_model=args.d_model, nhead=4,
        num_layers=args.layers, max_cells=32,
    )
    print(f"BrailleNet: {model.count_params():,} params")

    all_results = {}

    if args.stage in ['1', 'both']:
        t0 = time.time()
        results, best_acc = train_perception(
            model, device, n=args.n, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
            jitter=args.jitter, max_points=args.max_points,
            log_every=args.log_every,
        )
        elapsed = time.time() - t0
        print(f"\n  Stage 1 complete: best_cell_acc={best_acc:.4f} ({elapsed:.1f}s)")
        print(f"  {'PASSED ✓' if best_acc > 0.95 else 'NEEDS MORE TRAINING ✗'}")
        all_results['perception'] = {
            'best_cell_acc': best_acc,
            'elapsed': elapsed,
            'curve': results,
        }

    if args.stage in ['3.5', 'both']:
        t0 = time.time()
        results, best_vec, best_scl = train_algebra(
            model, device, n=args.n, epochs=args.epochs, lr=args.lr,
            batch_size=64, steps_per_epoch=args.steps_per_epoch,
        )
        elapsed = time.time() - t0
        print(f"\n  Stage 3.5 complete: best_vec={best_vec:.4f} best_scalar={best_scl:.4f} ({elapsed:.1f}s)")
        print(f"  {'PASSED ✓' if best_vec > 0.90 else 'NEEDS MORE TRAINING ✗'}")
        all_results['algebra'] = {
            'best_vec_acc': best_vec,
            'best_scalar_acc': best_scl,
            'elapsed': elapsed,
            'curve': results,
        }

    # Save model
    save_path = args.save or f'models/braillenet_n{args.n}.pth'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'n': args.n,
        'd_model': args.d_model,
        'num_layers': args.layers,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'curve'}
                    for k, v in all_results.items()},
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Save results
    results_path = Path('results') / f'braillenet_n{args.n}_train.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'curve'}
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
