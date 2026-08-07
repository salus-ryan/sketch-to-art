"""
BrailleNet foundation model — Modal H100 training.

Trains the stroke-native braille foundation model on Modal GPUs.
Much faster than local CPU/MPS training.

Stages:
  1:   Stroke → Cell perception
  3.5: Ternary state update algebra
  both: Sequential training of both

Usage:
    modal run train/modal_train_foundation.py
    modal run train/modal_train_foundation.py --stage 1 --epochs 100
    modal run train/modal_train_foundation.py --stage both --n 8 --epochs 50
"""

import modal

app = modal.App("braillenet-foundation")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1")
)

results_volume = modal.Volume.from_name("braillenet-results", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": results_volume},
)
def train(
    n: int = 8,
    stage: str = "both",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 128,
    num_layers: int = 4,
    steps_per_epoch: int = 500,
    jitter: str = "medium",
    max_points: int = 256,
):
    import json
    import math
    import random
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Config: n={n}, stage={stage}, epochs={epochs}, batch={batch_size}, "
          f"lr={lr}, d_model={d_model}, layers={num_layers}")

    # ===================================================================
    # INLINE: braille_data.py (data generation)
    # ===================================================================

    BRAILLE = {
        'a': [1], 'b': [1,2], 'c': [1,4], 'd': [1,4,5],
        'e': [1,5], 'f': [1,2,4], 'g': [1,2,4,5], 'h': [1,2,5],
        'i': [2,4], 'j': [2,4,5], 'k': [1,3], 'l': [1,2,3],
        'm': [1,3,4], 'n': [1,3,4,5], 'o': [1,3,5], 'p': [1,2,3,4],
        'q': [1,2,3,4,5], 'r': [1,2,3,5], 's': [2,3,4], 't': [2,3,4,5],
        'u': [1,3,6], 'v': [1,2,3,6], 'w': [2,4,5,6], 'x': [1,3,4,6],
        'y': [1,3,4,5,6], 'z': [1,3,5,6],
        'A': [1,7], 'B': [1,2,7], 'C': [1,4,7], 'D': [1,4,5,7],
        'E': [1,5,7], 'F': [1,2,4,7], 'G': [1,2,4,5,7], 'H': [1,2,5,7],
        'I': [2,4,7], 'J': [2,4,5,7], 'K': [1,3,7], 'L': [1,2,3,7],
        'M': [1,3,4,7], 'N': [1,3,4,5,7], 'O': [1,3,5,7], 'P': [1,2,3,4,7],
        'Q': [1,2,3,4,5,7], 'R': [1,2,3,5,7], 'S': [2,3,4,7], 'T': [2,3,4,5,7],
        'U': [1,3,6,7], 'V': [1,2,3,6,7], 'W': [2,4,5,6,7], 'X': [1,3,4,6,7],
        'Y': [1,3,4,5,6,7], 'Z': [1,3,5,6,7],
        '0': [2,4,5,8], '1': [1,8], '2': [1,2,8], '3': [1,4,8],
        '4': [1,4,5,8], '5': [1,5,8], '6': [1,2,4,8], '7': [1,2,4,5,8],
        '8': [1,2,5,8], '9': [2,4,8],
        ' ': [], '.': [2,5,6], ',': [2], '!': [2,3,5], '?': [2,3,6],
        ':': [2,5], '-': [3,6], '#': [3,4,5,6],
        '+': [3,4,6], '=': [4,6], '*': [1,6], '/': [3,4],
        '(': [1,2,3,5,6], ')': [2,3,4,5,6],
    }

    WORDS = [
        'hello', 'world', 'braille', 'dots', 'read', 'write', 'learn',
        'touch', 'feel', 'see', 'art', 'draw', 'sketch', 'line', 'curve',
        'math', 'code', 'data', 'text', 'sign', 'type', 'hand',
        'cat', 'dog', 'sun', 'moon', 'star', 'tree', 'home', 'book',
    ]

    _dot_pos_cache = {}
    def dot_positions(ndots):
        if ndots not in _dot_pos_cache:
            rows = math.ceil(ndots / 2)
            pos = {}
            for i in range(1, ndots + 1):
                col = 0 if i <= rows else 1
                row = (i - 1) if i <= rows else (i - rows - 1)
                pos[i] = (0.3 if col == 0 else 0.7, (row + 0.5) / rows)
            _dot_pos_cache[ndots] = pos
        return _dot_pos_cache[ndots]

    def dots_to_vector(dot_list, ndots):
        vec = [0] * ndots
        for d in dot_list:
            idx = abs(d) - 1
            if 0 <= idx < ndots:
                vec[idx] = 1 if d > 0 else -1
        return vec

    def vector_to_dots(vec):
        return {i + 1: int(v) for i, v in enumerate(vec) if v != 0}

    JITTER_CONFIGS = {
        'none': None,
        'light': {'position': 0.02, 'scale': 0.05, 'rotation': 0.05},
        'medium': {'position': 0.05, 'scale': 0.1, 'rotation': 0.1},
        'heavy': {'position': 0.1, 'scale': 0.15, 'rotation': 0.2},
    }

    def braille_cell_strokes(dots, ox, oy, cw, ch, ndots, jcfg=None):
        strokes = []
        positions = dot_positions(ndots)
        dot_r = min(cw, ch) * (0.6 / math.ceil(ndots / 2))
        segments = 8

        if isinstance(dots, list):
            signed_dots = {abs(d): (1 if d > 0 else -1) for d in dots}
        else:
            signed_dots = dict(dots)

        if jcfg:
            ox += random.uniform(-jcfg['position'], jcfg['position']) * cw
            oy += random.uniform(-jcfg['position'], jcfg['position']) * ch
            s = 1.0 + random.uniform(-jcfg['scale'], jcfg['scale'])
            cw *= s; ch *= s; dot_r *= s
            theta = random.uniform(-jcfg['rotation'], jcfg['rotation'])
        else:
            theta = 0

        for d_key, value in signed_dots.items():
            d = int(d_key) if isinstance(d_key, str) else d_key
            if d not in positions or value == 0:
                continue
            dx, dy = positions[d]
            cx = ox + dx * cw
            cy = oy + dy * ch

            if theta != 0:
                ccx, ccy = ox + 0.5 * cw, oy + 0.5 * ch
                rx, ry = cx - ccx, cy - ccy
                cx = ccx + rx * math.cos(theta) - ry * math.sin(theta)
                cy = ccy + rx * math.sin(theta) + ry * math.cos(theta)

            if value > 0:
                pts = []
                for i in range(segments + 1):
                    a = (i / segments) * math.pi * 2
                    px = cx + math.cos(a) * dot_r
                    py = cy + math.sin(a) * dot_r
                    if jcfg:
                        px += random.gauss(0, jcfg['position'] * dot_r * 0.3)
                        py += random.gauss(0, jcfg['position'] * dot_r * 0.3)
                    pts.append((px, py))
                strokes.append(pts)
            else:
                r = dot_r * 0.9
                strokes.append([(cx - r, cy - r), (cx + r, cy + r)])
                strokes.append([(cx + r, cy - r), (cx - r, cy + r)])
        return strokes

    def cell_add(a, b):
        return [max(-1, min(1, ai + bi)) for ai, bi in zip(a, b)]

    def cell_negate(a):
        return [-v for v in a]

    def cell_inner(a, b):
        return sum(ai * bi for ai, bi in zip(a, b))

    def cell_update(memory, delta):
        return cell_add(memory, delta)

    # --- Stroke encoding ---
    SEP_TOKEN = [-1.0, -1.0]
    PAD_TOKEN = [-2.0, -2.0]

    def encode_strokes(strokes, mp):
        flat = []
        for i, stroke in enumerate(strokes):
            if i > 0:
                flat.append(SEP_TOKEN)
            for x, y in stroke:
                flat.append([x, y])
        if len(flat) > mp:
            flat = flat[:mp]
        num_real = len(flat)
        while len(flat) < mp:
            flat.append(PAD_TOKEN)
        points = torch.tensor(flat, dtype=torch.float32)
        mask = torch.zeros(mp, dtype=torch.bool)
        mask[:num_real] = True
        return points, mask

    def encode_cells(cells, ndots, mc=32):
        num_cells = min(len(cells), mc)
        ct = torch.zeros(mc, ndots, dtype=torch.float32)
        cm = torch.zeros(mc, dtype=torch.bool)
        for i in range(num_cells):
            ct[i] = torch.tensor(cells[i], dtype=torch.float32)
            cm[i] = True
        return ct, cm

    # --- Sample generators ---
    def gen_text_sample(ndots, jcfg):
        text = random.choice(WORDS)
        if random.random() < 0.2:
            text = text.upper()
        elif random.random() < 0.3:
            text = text.capitalize()

        strokes = []
        cells = []
        spacing = 0.07 * 1.3
        for i, ch in enumerate(text):
            dl = BRAILLE.get(ch)
            if dl is None:
                continue
            cells.append(dots_to_vector(dl, ndots))
            if dl:
                strokes.extend(braille_cell_strokes(
                    dl, 0.05 + i * spacing, 0.25, 0.07, 0.16, ndots, jcfg))
        return strokes, cells

    def gen_raw_cell_sample(ndots, jcfg):
        vec = [random.choice([-1, 0, 1]) for _ in range(ndots)]
        dots = vector_to_dots(vec)
        ox = random.uniform(0.1, 0.6)
        oy = random.uniform(0.1, 0.6)
        cw = random.uniform(0.08, 0.15)
        ch = random.uniform(0.12, 0.25)
        strokes = braille_cell_strokes(dots, ox, oy, cw, ch, ndots, jcfg)
        return strokes, [vec]

    def gen_algebra_sample(ndots):
        a = [random.choice([-1, 0, 1]) for _ in range(ndots)]
        b = [random.choice([-1, 0, 1]) for _ in range(ndots)]
        op = random.choice(['add', 'negate', 'inner', 'cancel', 'update'])
        if op == 'add':
            result, rt = cell_add(a, b), 'vector'
        elif op == 'negate':
            result, rt = cell_negate(a), 'vector'
        elif op == 'inner':
            result, rt = cell_inner(a, b), 'scalar'
        elif op == 'cancel':
            b = cell_negate(a)
            result, rt = [0] * ndots, 'vector'
        else:
            result, rt = cell_update(a, b), 'vector'
        return a, b, op, result, rt

    OP_NAMES = ['add', 'negate', 'inner', 'cancel', 'update']

    # ===================================================================
    # INLINE: braille_model.py (model architecture)
    # ===================================================================

    class StrokeEncoder(nn.Module):
        def __init__(self, dm=128, nhead=4, nlayers=4, mc=32, dropout=0.1):
            super().__init__()
            self.dm = dm
            self.mc = mc
            self.point_proj = nn.Sequential(
                nn.Linear(4, dm), nn.GELU(), nn.Linear(dm, dm))
            self.pos_enc = nn.Parameter(torch.randn(1, max_points, dm) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dm, nhead=nhead, dim_feedforward=dm * 4,
                dropout=dropout, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
            self.cell_assign = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, mc))
            self.norm = nn.LayerNorm(dm)

        def forward(self, points, mask):
            B, T, _ = points.shape
            is_sep = ((points[:,:,0]==-1)&(points[:,:,1]==-1)).float().unsqueeze(-1)
            is_pad = ((points[:,:,0]==-2)&(points[:,:,1]==-2)).float().unsqueeze(-1)
            xy = points.clone()
            xy[~mask] = 0.0
            xy[(is_sep.squeeze(-1) > 0.5)] = 0.0
            features = torch.cat([xy, is_sep, is_pad], dim=-1)
            h = self.point_proj(features) + self.pos_enc[:, :T, :]
            h = self.transformer(h, src_key_padding_mask=~mask)
            h = self.norm(h)
            cl = self.cell_assign(h)
            cl = cl.masked_fill(~mask.unsqueeze(-1), -1e9)
            cw = F.softmax(cl, dim=1)
            cell_emb = torch.bmm(cw.transpose(1,2), h)
            cell_mass = cw.sum(dim=1)
            return cell_emb, cell_mass > 0.1

    class CellDecoder(nn.Module):
        def __init__(self, dm=128, ndots=8):
            super().__init__()
            self.ndots = ndots
            self.dec = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, ndots * 3))

        def forward(self, cell_emb):
            B, C, _ = cell_emb.shape
            return self.dec(cell_emb).view(B, C, self.ndots, 3)

    class AlgebraHead(nn.Module):
        def __init__(self, ndots=8, nops=5, hidden=256):
            super().__init__()
            self.ndots = ndots
            inp = ndots + ndots + nops
            self.shared = nn.Sequential(
                nn.Linear(inp, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU())
            self.vec_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, ndots * 3))
            self.scl_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, 1))

        def forward(self, ca, cb, op):
            h = self.shared(torch.cat([ca, cb, op], dim=-1))
            return self.vec_head(h).view(-1, self.ndots, 3), self.scl_head(h)

    class BrailleNet(nn.Module):
        def __init__(self, ndots=8, dm=128, nhead=4, nlayers=4, mc=32, dropout=0.1):
            super().__init__()
            self.ndots = ndots
            self.encoder = StrokeEncoder(dm, nhead, nlayers, mc, dropout)
            self.decoder = CellDecoder(dm, ndots)
            self.algebra = AlgebraHead(ndots, 5, dm * 2)

        def perceive(self, points, mask):
            ce, cm = self.encoder(points, mask)
            return self.decoder(ce), cm

        def count_params(self):
            return sum(p.numel() for p in self.parameters())

    # ===================================================================
    # LOSS + METRICS
    # ===================================================================

    def perception_loss(dot_logits, target_cells, cell_mask):
        target_classes = (target_cells + 1).long()
        lf = dot_logits[cell_mask]
        tf = target_classes[cell_mask]
        if lf.shape[0] == 0:
            return torch.tensor(0.0, device=dot_logits.device), \
                   {'dot_acc': 0.0, 'cell_acc': 0.0, 'num_cells': 0}
        loss = F.cross_entropy(lf.reshape(-1, 3), tf.reshape(-1))
        with torch.no_grad():
            pred = lf.argmax(dim=-1)
            dc = (pred == tf).float()
            da = dc.mean().item()
            ca = dc.all(dim=-1).float().mean().item()
        return loss, {'dot_acc': da, 'cell_acc': ca, 'num_cells': lf.shape[0]}

    def algebra_loss_fn(vl, sp, tr, is_scl):
        B, ndots, _ = vl.shape
        is_vec = 1.0 - is_scl
        tc = (tr + 1).long()
        vec_l = F.cross_entropy(vl.reshape(-1, 3), tc.reshape(-1),
                                reduction='none').view(B, ndots)
        vec_l = (vec_l.mean(dim=1) * is_vec).sum() / max(is_vec.sum(), 1)
        st = tr[:, 0:1]
        scl_l = F.mse_loss(sp, st, reduction='none').squeeze(-1)
        scl_l = (scl_l * is_scl).sum() / max(is_scl.sum(), 1)
        loss = vec_l + scl_l
        with torch.no_grad():
            pv = vl.argmax(dim=-1)
            correct = (pv == tc)
            vda = correct[is_vec > 0.5].float().mean().item() if is_vec.sum() > 0 else 0.0
            vca = correct[is_vec > 0.5].all(dim=-1).float().mean().item() if is_vec.sum() > 0 else 0.0
            if is_scl.sum() > 0:
                err = (sp.squeeze(-1) - st.squeeze(-1)).abs()
                sa = (err[is_scl > 0.5] < 0.5).float().mean().item()
            else:
                sa = 0.0
        return loss, {'vda': vda, 'vca': vca, 'sa': sa}

    # ===================================================================
    # BATCH GENERATION (on-the-fly, GPU-side)
    # ===================================================================

    def gen_perception_batch(bs, ndots, mp, mc, jcfg, dev):
        all_pts, all_masks, all_cells, all_cmasks = [], [], [], []
        for _ in range(bs):
            if random.random() < 0.5:
                strokes, cells = gen_text_sample(ndots, jcfg)
            else:
                strokes, cells = gen_raw_cell_sample(ndots, jcfg)
            if not strokes or not cells:
                cells = [[0]*ndots]
                strokes = [[(0.5, 0.5), (0.5, 0.5)]]
            pts, mask = encode_strokes(strokes, mp)
            ct, cm = encode_cells(cells, ndots, mc)
            all_pts.append(pts)
            all_masks.append(mask)
            all_cells.append(ct)
            all_cmasks.append(cm)
        return (torch.stack(all_pts).to(dev), torch.stack(all_masks).to(dev),
                torch.stack(all_cells).to(dev), torch.stack(all_cmasks).to(dev))

    def gen_algebra_batch(bs, ndots, dev):
        cas, cbs, ops, results, is_scls = [], [], [], [], []
        for _ in range(bs):
            a, b, op, res, rt = gen_algebra_sample(ndots)
            cas.append(a)
            cbs.append(b)
            oh = [0.0] * len(OP_NAMES)
            oh[OP_NAMES.index(op)] = 1.0
            ops.append(oh)
            if rt == 'vector':
                results.append(res)
                is_scls.append(0.0)
            else:
                results.append([res / ndots] * ndots)
                is_scls.append(1.0)
        return (torch.tensor(cas, dtype=torch.float32, device=dev),
                torch.tensor(cbs, dtype=torch.float32, device=dev),
                torch.tensor(ops, dtype=torch.float32, device=dev),
                torch.tensor(results, dtype=torch.float32, device=dev),
                torch.tensor(is_scls, dtype=torch.float32, device=dev))

    # ===================================================================
    # TRAINING
    # ===================================================================

    model = BrailleNet(ndots=n, dm=d_model, nhead=4, nlayers=num_layers, mc=32).to(device)
    total_params = model.count_params()
    print(f"BrailleNet: {total_params:,} params")

    jcfg = JITTER_CONFIGS.get(jitter)
    all_results = {}

    # --- Stage 1: Perception ---
    if stage in ['1', 'both']:
        print(f"\n{'='*60}")
        print(f"STAGE 1: Stroke → Cell Perception (H100)")
        print(f"  n={n}, jitter={jitter}, lr={lr}, batch={batch_size}")
        print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs")
        print(f"{'='*60}")

        opt = torch.optim.AdamW(
            list(model.encoder.parameters()) + list(model.decoder.parameters()),
            lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * steps_per_epoch)

        best_cell_acc = 0.0
        perc_curve = []
        t0 = time.time()

        for epoch in range(epochs):
            model.train()
            ep_loss, ep_da, ep_ca, ep_nc = 0, 0, 0, 0

            for step in range(steps_per_epoch):
                pts, msk, tgt, cmsk = gen_perception_batch(
                    batch_size, n, max_points, 32, jcfg, device)

                dl, _ = model.perceive(pts, msk)
                loss, met = perception_loss(dl, tgt, cmsk)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                ep_loss += loss.item()
                ep_da += met['dot_acc'] * met['num_cells']
                ep_ca += met['cell_acc'] * met['num_cells']
                ep_nc += met['num_cells']

            # Eval on clean data
            model.eval()
            ev_da, ev_ca, ev_nc = 0, 0, 0
            with torch.no_grad():
                for _ in range(100):
                    pts, msk, tgt, cmsk = gen_perception_batch(
                        batch_size, n, max_points, 32, None, device)
                    dl, _ = model.perceive(pts, msk)
                    _, met = perception_loss(dl, tgt, cmsk)
                    ev_da += met['dot_acc'] * met['num_cells']
                    ev_ca += met['cell_acc'] * met['num_cells']
                    ev_nc += met['num_cells']

            eval_dot = ev_da / max(ev_nc, 1)
            eval_cell = ev_ca / max(ev_nc, 1)
            best_cell_acc = max(best_cell_acc, eval_cell)
            train_dot = ep_da / max(ep_nc, 1)
            train_cell = ep_ca / max(ep_nc, 1)

            row = {
                'epoch': epoch,
                'train_loss': round(ep_loss / steps_per_epoch, 6),
                'train_dot': round(train_dot, 4),
                'train_cell': round(train_cell, 4),
                'eval_dot': round(eval_dot, 4),
                'eval_cell': round(eval_cell, 4),
                'best': round(best_cell_acc, 4),
            }
            perc_curve.append(row)

            if epoch % 5 == 0 or epoch == epochs - 1:
                elapsed = time.time() - t0
                print(f"  E{epoch:3d} | loss={row['train_loss']:.4f} | "
                      f"train d={train_dot:.4f} c={train_cell:.4f} | "
                      f"eval d={eval_dot:.4f} c={eval_cell:.4f} | "
                      f"best={best_cell_acc:.4f} | {elapsed:.0f}s")

        perc_elapsed = time.time() - t0
        passed = best_cell_acc > 0.95
        print(f"\n  Stage 1: best_cell_acc={best_cell_acc:.4f} ({perc_elapsed:.1f}s)")
        print(f"  {'PASSED ✓' if passed else 'NEEDS MORE TRAINING ✗'}")
        all_results['perception'] = {
            'best_cell_acc': best_cell_acc,
            'elapsed': perc_elapsed,
            'passed': passed,
            'curve': perc_curve,
        }

    # --- Stage 3.5: Algebra ---
    if stage in ['3.5', 'both']:
        print(f"\n{'='*60}")
        print(f"STAGE 3.5: Ternary State Update Algebra (H100)")
        print(f"  n={n}, lr={lr}, batch={batch_size}")
        print(f"  Operations: add, negate, inner, cancel, update")
        print(f"{'='*60}")

        opt = torch.optim.AdamW(model.algebra.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * steps_per_epoch)

        best_vec, best_scl = 0.0, 0.0
        alg_curve = []
        t0 = time.time()

        for epoch in range(epochs):
            model.train()
            ep_loss = 0
            ep_vda, ep_vca, ep_sa = 0, 0, 0
            n_vec, n_scl = 0, 0

            for step in range(steps_per_epoch):
                ca, cb, op, tgt, is_s = gen_algebra_batch(batch_size, n, device)
                vl, sp = model.algebra(ca, cb, op)
                loss, met = algebra_loss_fn(vl, sp, tgt, is_s)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()

                ep_loss += loss.item()
                bv = (is_s < 0.5).sum().item()
                bs_ = (is_s > 0.5).sum().item()
                ep_vda += met['vda'] * bv
                ep_vca += met['vca'] * bv
                ep_sa += met['sa'] * bs_
                n_vec += bv
                n_scl += bs_

            avg_vda = ep_vda / max(n_vec, 1)
            avg_vca = ep_vca / max(n_vec, 1)
            avg_sa = ep_sa / max(n_scl, 1)
            best_vec = max(best_vec, avg_vca)
            best_scl = max(best_scl, avg_sa)

            row = {
                'epoch': epoch,
                'loss': round(ep_loss / steps_per_epoch, 6),
                'vda': round(avg_vda, 4), 'vca': round(avg_vca, 4),
                'sa': round(avg_sa, 4),
            }
            alg_curve.append(row)

            if epoch % 5 == 0 or epoch == epochs - 1:
                elapsed = time.time() - t0
                print(f"  E{epoch:3d} | loss={row['loss']:.4f} | "
                      f"vec_dot={avg_vda:.4f} vec_cell={avg_vca:.4f} | "
                      f"scalar={avg_sa:.4f} | {elapsed:.0f}s")

        alg_elapsed = time.time() - t0
        print(f"\n  Stage 3.5: best_vec={best_vec:.4f} best_scalar={best_scl:.4f} "
              f"({alg_elapsed:.1f}s)")
        print(f"  {'PASSED ✓' if best_vec > 0.90 else 'NEEDS MORE TRAINING ✗'}")
        all_results['algebra'] = {
            'best_vec_acc': best_vec,
            'best_scalar_acc': best_scl,
            'elapsed': alg_elapsed,
            'passed': best_vec > 0.90,
            'curve': alg_curve,
        }

    # ===================================================================
    # SAVE
    # ===================================================================

    # Save model to volume
    model_path = f"/results/braillenet_n{n}.pth"
    torch.save({
        'model_state': model.state_dict(),
        'n': n, 'd_model': d_model, 'num_layers': num_layers,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'curve'}
                    for k, v in all_results.items()},
    }, model_path)

    # Save results JSON
    results_path = f"/results/braillenet_n{n}_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    results_volume.commit()
    print(f"\nSaved model to {model_path}")
    print(f"Saved results to {results_path}")

    return all_results


# ===================================================================
# CLI ENTRYPOINT
# ===================================================================

@app.local_entrypoint()
def main(
    n: int = 8,
    stage: str = "both",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    steps_per_epoch: int = 500,
    jitter: str = "medium",
):
    results = train.remote(
        n=n, stage=stage, epochs=epochs, batch_size=batch_size,
        lr=lr, steps_per_epoch=steps_per_epoch, jitter=jitter,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for stage_name, res in results.items():
        print(f"\n{stage_name}:")
        for k, v in res.items():
            if k != 'curve':
                print(f"  {k}: {v}")
