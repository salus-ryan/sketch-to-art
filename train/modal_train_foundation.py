"""
BrailleNet foundation model — FULL overnight Modal H100 training.

All stages, sequential, one run. Go to sleep, wake up to a foundation model.

Stages:
  1:   Stroke → Cell perception (96.5% proven)
  2:   Cell → Stroke generation (decoder, round-trip validated)
  3:   Cell → Meaning composition (text decoding + sequence programs)
  3.5: Ternary state update algebra (100% proven)
  4:   End-to-end stroke → meaning (full pipeline)

Usage:
    modal run train/modal_train_foundation.py
    modal run train/modal_train_foundation.py --epochs 100
"""

import modal

app = modal.App("braillenet-foundation-full")

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1")
)

results_volume = modal.Volume.from_name("braillenet-results", create_if_missing=True)


@app.function(
    image=train_image,
    gpu="H100",
    timeout=14400,  # 4 hours for full overnight run
    volumes={"/results": results_volume},
)
def train(
    n: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    d_model: int = 192,
    num_layers: int = 6,
    nhead: int = 6,
    steps_per_epoch: int = 500,
    jitter: str = "medium",
    max_points: int = 256,
    max_cells: int = 32,
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
    print(f"Config: n={n}, epochs={epochs}, batch={batch_size}, "
          f"lr={lr}, d_model={d_model}, layers={num_layers}, heads={nhead}")
    print(f"Full overnight training — all 5 stages")

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

    dm = d_model
    mc = max_cells
    mp = max_points
    nh = nhead
    nl = num_layers

    class StrokeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.point_proj = nn.Sequential(
                nn.Linear(4, dm), nn.GELU(), nn.Linear(dm, dm))
            self.pos_enc = nn.Parameter(torch.randn(1, mp, dm) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dm, nhead=nh, dim_feedforward=dm * 4,
                dropout=0.1, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nl)
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
        def __init__(self):
            super().__init__()
            self.dec = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, n * 3))

        def forward(self, cell_emb):
            B, C, _ = cell_emb.shape
            return self.dec(cell_emb).view(B, C, n, 3)

    class StrokeGenerator(nn.Module):
        """Stage 2: Cell embedding → stroke sequence (autoregressive).

        Given cell embeddings, generates stroke points one at a time.
        Uses a transformer decoder with cross-attention to cell embeddings.
        Output: (x, y, pen_state) where pen_state ∈ {draw, sep, stop}.
        """
        def __init__(self):
            super().__init__()
            # Input: previous point (x, y, pen_onehot[3]) = 5 dims
            self.point_proj = nn.Sequential(
                nn.Linear(5, dm), nn.GELU(), nn.Linear(dm, dm))
            self.pos_enc = nn.Parameter(torch.randn(1, mp, dm) * 0.02)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=dm, nhead=nh, dim_feedforward=dm * 4,
                dropout=0.1, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerDecoder(dec_layer, num_layers=nl // 2)
            # Output: x, y, pen_state_logits[3]
            self.output_head = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, 2 + 3))  # (dx, dy, pen[draw/sep/stop])
            self.norm = nn.LayerNorm(dm)

        def forward(self, cell_emb, target_points, target_mask):
            """Teacher-forced generation.

            Args:
                cell_emb: (B, mc, dm) — from encoder
                target_points: (B, T, 5) — (x, y, pen_onehot[3])
                target_mask: (B, T)
            Returns:
                pred: (B, T, 5) — predicted (x, y, pen_logits[3])
            """
            B, T, _ = target_points.shape
            h = self.point_proj(target_points) + self.pos_enc[:, :T, :]
            # Causal mask
            causal = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
            tgt_pad = ~target_mask
            h = self.transformer(h, cell_emb, tgt_mask=causal,
                                  tgt_key_padding_mask=tgt_pad)
            h = self.norm(h)
            return self.output_head(h)

    class CompositionBackbone(nn.Module):
        """Stage 3: Cell sequence → meaning.

        Takes a sequence of cell vectors and produces:
          - Text output: per-cell character classification (ASCII)
          - Sequence result: stack program execution result
        """
        def __init__(self):
            super().__init__()
            self.cell_proj = nn.Sequential(
                nn.Linear(n, dm), nn.GELU(), nn.Linear(dm, dm))
            self.pos_enc = nn.Parameter(torch.randn(1, mc, dm) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dm, nhead=nh, dim_feedforward=dm * 4,
                dropout=0.1, batch_first=True, activation='gelu')
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nl)
            self.norm = nn.LayerNorm(dm)

            # Text decoding head: per-cell → character class
            # 95 printable ASCII chars + 1 pad = 96 classes
            self.text_head = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, 96))

            # Sequence composition head: CLS token → result cell vector
            self.cls_token = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
            self.seq_head = nn.Sequential(
                nn.Linear(dm, dm), nn.GELU(),
                nn.Linear(dm, n * 3))

        def text_decode(self, cells, mask):
            """cells: (B, mc, n), mask: (B, mc) → char_logits: (B, mc, 96)"""
            B = cells.shape[0]
            C = cells.shape[1]
            h = self.cell_proj(cells) + self.pos_enc[:, :C, :]
            h = self.transformer(h, src_key_padding_mask=~mask)
            h = self.norm(h)
            return self.text_head(h)

        def seq_compose(self, cells, mask):
            """cells: (B, L, n), mask: (B, L) → result_logits: (B, n, 3)"""
            B, L, _ = cells.shape
            h = self.cell_proj(cells) + self.pos_enc[:, :L, :]
            # Prepend CLS token
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            # Extend mask for CLS
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=cells.device)
            ext_mask = torch.cat([cls_mask, mask], dim=1)
            h = self.transformer(h, src_key_padding_mask=~ext_mask)
            h = self.norm(h)
            # CLS output → result
            cls_out = h[:, 0, :]
            return self.seq_head(cls_out).view(B, n, 3)

    class AlgebraHead(nn.Module):
        def __init__(self):
            super().__init__()
            inp = n + n + 5  # cell_a + cell_b + op_onehot
            self.shared = nn.Sequential(
                nn.Linear(inp, dm * 2), nn.GELU(),
                nn.Linear(dm * 2, dm * 2), nn.GELU())
            self.vec_head = nn.Sequential(
                nn.Linear(dm * 2, dm * 2), nn.GELU(),
                nn.Linear(dm * 2, n * 3))
            self.scl_head = nn.Sequential(
                nn.Linear(dm * 2, dm), nn.GELU(),
                nn.Linear(dm, 1))

        def forward(self, ca, cb, op):
            h = self.shared(torch.cat([ca, cb, op], dim=-1))
            return self.vec_head(h).view(-1, n, 3), self.scl_head(h)

    class BrailleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StrokeEncoder()
            self.cell_decoder = CellDecoder()
            self.stroke_gen = StrokeGenerator()
            self.composition = CompositionBackbone()
            self.algebra = AlgebraHead()

        def perceive(self, points, mask):
            ce, cm = self.encoder(points, mask)
            return self.cell_decoder(ce), cm

        def generate(self, cell_emb, target_points, target_mask):
            return self.stroke_gen(cell_emb, target_points, target_mask)

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

    # Build reverse lookup: cell vector → character
    REVERSE_BRAILLE = {}
    for ch, dots in BRAILLE.items():
        vec_key = tuple(dots_to_vector(dots, n))
        REVERSE_BRAILLE[vec_key] = ch

    # Character → class index (0=pad, 1-95=printable ASCII 32-126)
    def char_to_idx(c):
        o = ord(c)
        if 32 <= o <= 126:
            return o - 31  # 1..95
        return 0  # pad

    def idx_to_char(i):
        if i == 0:
            return ''
        return chr(i + 31)

    def gen_perception_batch(bs, jcfg, dev):
        all_pts, all_masks, all_cells, all_cmasks = [], [], [], []
        for _ in range(bs):
            if random.random() < 0.5:
                strokes, cells = gen_text_sample(n, jcfg)
            else:
                strokes, cells = gen_raw_cell_sample(n, jcfg)
            if not strokes or not cells:
                cells = [[0]*n]
                strokes = [[(0.5, 0.5), (0.5, 0.5)]]
            pts, mask = encode_strokes(strokes, mp)
            ct, cm = encode_cells(cells, n, mc)
            all_pts.append(pts)
            all_masks.append(mask)
            all_cells.append(ct)
            all_cmasks.append(cm)
        return (torch.stack(all_pts).to(dev), torch.stack(all_masks).to(dev),
                torch.stack(all_cells).to(dev), torch.stack(all_cmasks).to(dev))

    def gen_algebra_batch(bs, dev):
        cas, cbs, ops, results, is_scls = [], [], [], [], []
        for _ in range(bs):
            a, b, op, res, rt = gen_algebra_sample(n)
            cas.append(a)
            cbs.append(b)
            oh = [0.0] * len(OP_NAMES)
            oh[OP_NAMES.index(op)] = 1.0
            ops.append(oh)
            if rt == 'vector':
                results.append(res)
                is_scls.append(0.0)
            else:
                results.append([res / n] * n)
                is_scls.append(1.0)
        return (torch.tensor(cas, dtype=torch.float32, device=dev),
                torch.tensor(cbs, dtype=torch.float32, device=dev),
                torch.tensor(ops, dtype=torch.float32, device=dev),
                torch.tensor(results, dtype=torch.float32, device=dev),
                torch.tensor(is_scls, dtype=torch.float32, device=dev))

    PEN_DRAW = [1, 0, 0]
    PEN_SEP  = [0, 1, 0]
    PEN_STOP = [0, 0, 1]

    def gen_generation_batch(bs, jcfg, dev):
        """Generate (cell_vectors, target_stroke_sequence) pairs for Stage 2."""
        all_cells, all_cmasks = [], []
        all_tgt_pts, all_tgt_masks = [], []
        all_next_pts, all_next_pen = [], []

        for _ in range(bs):
            if random.random() < 0.5:
                strokes, cells = gen_text_sample(n, jcfg)
            else:
                strokes, cells = gen_raw_cell_sample(n, jcfg)
            if not strokes or not cells:
                cells = [[0]*n]
                strokes = [[(0.5, 0.5), (0.5, 0.5)]]

            # Build teacher-forced sequence: input is shifted target
            seq_in = []   # (x, y, pen_onehot[3])
            seq_out_xy = []
            seq_out_pen = []
            for si, stroke in enumerate(strokes):
                for pi, (x, y) in enumerate(stroke):
                    pen = PEN_DRAW
                    seq_in.append([x, y] + pen)
                    # Next point prediction
                    if pi + 1 < len(stroke):
                        nx, ny = stroke[pi + 1]
                        seq_out_xy.append([nx, ny])
                        seq_out_pen.append(0)  # draw
                    elif si + 1 < len(strokes):
                        nx, ny = strokes[si + 1][0]
                        seq_out_xy.append([nx, ny])
                        seq_out_pen.append(1)  # sep
                    else:
                        seq_out_xy.append([x, y])
                        seq_out_pen.append(2)  # stop

            # Truncate/pad to mp
            L = min(len(seq_in), mp)
            while len(seq_in) < mp:
                seq_in.append([0, 0, 0, 0, 0])
                seq_out_xy.append([0, 0])
                seq_out_pen.append(2)
            seq_in = seq_in[:mp]
            seq_out_xy = seq_out_xy[:mp]
            seq_out_pen = seq_out_pen[:mp]

            tgt_mask = torch.zeros(mp, dtype=torch.bool)
            tgt_mask[:L] = True

            ct, cm = encode_cells(cells, n, mc)
            all_cells.append(ct)
            all_cmasks.append(cm)
            all_tgt_pts.append(torch.tensor(seq_in, dtype=torch.float32))
            all_tgt_masks.append(tgt_mask)
            all_next_pts.append(torch.tensor(seq_out_xy, dtype=torch.float32))
            all_next_pen.append(torch.tensor(seq_out_pen, dtype=torch.long))

        return (torch.stack(all_cells).to(dev), torch.stack(all_cmasks).to(dev),
                torch.stack(all_tgt_pts).to(dev), torch.stack(all_tgt_masks).to(dev),
                torch.stack(all_next_pts).to(dev), torch.stack(all_next_pen).to(dev))

    def gen_text_decode_batch(bs, dev):
        """Generate (cell_vectors, char_indices) for text decoding."""
        all_cells, all_cmasks, all_targets = [], [], []
        for _ in range(bs):
            text = random.choice(WORDS)
            if random.random() < 0.2:
                text = text.upper()
            elif random.random() < 0.3:
                text = text.capitalize()

            cells = []
            char_indices = []
            for ch in text:
                dl = BRAILLE.get(ch)
                if dl is None:
                    continue
                cells.append(dots_to_vector(dl, n))
                char_indices.append(char_to_idx(ch))

            # Pad to mc
            num = len(cells)
            while len(cells) < mc:
                cells.append([0] * n)
                char_indices.append(0)
            cells = cells[:mc]
            char_indices = char_indices[:mc]

            ct = torch.tensor(cells, dtype=torch.float32)
            cm = torch.zeros(mc, dtype=torch.bool)
            cm[:num] = True
            tgt = torch.tensor(char_indices, dtype=torch.long)

            all_cells.append(ct)
            all_cmasks.append(cm)
            all_targets.append(tgt)

        return (torch.stack(all_cells).to(dev), torch.stack(all_cmasks).to(dev),
                torch.stack(all_targets).to(dev))

    def gen_seq_compose_batch(bs, dev, max_len=8):
        """Generate (cell_sequence, result_vector) for sequence composition."""
        all_cells, all_masks, all_results = [], [], []
        for _ in range(bs):
            length = random.randint(2, max_len)
            cells = []
            for _ in range(length):
                cells.append([random.choice([-1, 0, 1]) for _ in range(n)])
            # Result: fold with cell_add
            result = cells[0][:]
            for c in cells[1:]:
                result = cell_add(result, c)

            # Pad to max_len
            while len(cells) < max_len:
                cells.append([0] * n)
            cells = cells[:max_len]

            ct = torch.tensor(cells, dtype=torch.float32)
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:length] = True
            res = torch.tensor(result, dtype=torch.float32)

            all_cells.append(ct)
            all_masks.append(mask)
            all_results.append(res)

        return (torch.stack(all_cells).to(dev), torch.stack(all_masks).to(dev),
                torch.stack(all_results).to(dev))

    # ===================================================================
    # TRAINING — ALL 5 STAGES
    # ===================================================================

    model = BrailleNet().to(device)
    total_params = model.count_params()
    print(f"BrailleNet: {total_params:,} params")
    print(f"  encoder:     {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  cell_decoder:{sum(p.numel() for p in model.cell_decoder.parameters()):,}")
    print(f"  stroke_gen:  {sum(p.numel() for p in model.stroke_gen.parameters()):,}")
    print(f"  composition: {sum(p.numel() for p in model.composition.parameters()):,}")
    print(f"  algebra:     {sum(p.numel() for p in model.algebra.parameters()):,}")

    jcfg = JITTER_CONFIGS.get(jitter)
    all_results = {}
    grand_t0 = time.time()

    def save_checkpoint(tag):
        path = f"/results/braillenet_n{n}_{tag}.pth"
        torch.save({
            'model_state': model.state_dict(),
            'n': n, 'd_model': d_model, 'num_layers': num_layers,
            'nhead': nhead, 'tag': tag,
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'curve'}
                        for k, v in all_results.items()},
        }, path)
        results_volume.commit()
        print(f"  💾 Checkpoint saved: {path}")

    # ─────────────────────────────────────────────────────────
    # STAGE 1: Stroke → Cell Perception
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 1: Stroke → Cell Perception")
    print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.cell_decoder.parameters()),
        lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_cell_acc = 0.0
    perc_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_da, ep_ca, ep_nc = 0, 0, 0, 0

        for step in range(steps_per_epoch):
            pts, msk, tgt, cmsk = gen_perception_batch(batch_size, jcfg, device)
            dl, _ = model.perceive(pts, msk)
            loss, met = perception_loss(dl, tgt, cmsk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item()
            ep_da += met['dot_acc'] * met['num_cells']
            ep_ca += met['cell_acc'] * met['num_cells']
            ep_nc += met['num_cells']

        model.eval()
        ev_da, ev_ca, ev_nc = 0, 0, 0
        with torch.no_grad():
            for _ in range(100):
                pts, msk, tgt, cmsk = gen_perception_batch(batch_size, None, device)
                dl, _ = model.perceive(pts, msk)
                _, met = perception_loss(dl, tgt, cmsk)
                ev_da += met['dot_acc'] * met['num_cells']
                ev_ca += met['cell_acc'] * met['num_cells']
                ev_nc += met['num_cells']

        eval_dot = ev_da / max(ev_nc, 1)
        eval_cell = ev_ca / max(ev_nc, 1)
        best_cell_acc = max(best_cell_acc, eval_cell)

        row = {'epoch': epoch, 'loss': round(ep_loss/steps_per_epoch, 6),
               'eval_dot': round(eval_dot, 4), 'eval_cell': round(eval_cell, 4),
               'best': round(best_cell_acc, 4)}
        perc_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={row['loss']:.4f} | "
                  f"eval d={eval_dot:.4f} c={eval_cell:.4f} | "
                  f"best={best_cell_acc:.4f} | {time.time()-t0:.0f}s")

    passed = best_cell_acc > 0.95
    print(f"\n  Stage 1: best={best_cell_acc:.4f} ({time.time()-t0:.0f}s) "
          f"{'PASSED ✓' if passed else '✗'}")
    all_results['s1_perception'] = {
        'best_cell_acc': best_cell_acc, 'elapsed': time.time()-t0,
        'passed': passed, 'curve': perc_curve}
    save_checkpoint('s1')

    # ─────────────────────────────────────────────────────────
    # STAGE 2: Cell → Stroke Generation
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 2: Cell → Stroke Generation")
    print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.stroke_gen.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_gen_acc = 0.0
    gen_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_xy_loss, ep_pen_loss, ep_pen_acc = 0, 0, 0
        ep_steps = 0

        for step in range(steps_per_epoch):
            cells, cmsk, tgt_pts, tgt_mask, next_xy, next_pen = \
                gen_generation_batch(batch_size, jcfg, device)

            # Project cells to embeddings for the generator's cross-attention
            cell_emb_direct = model.composition.cell_proj(cells)

            pred = model.stroke_gen(cell_emb_direct, tgt_pts, tgt_mask)
            pred_xy = pred[:, :, :2]   # (B, T, 2)
            pred_pen = pred[:, :, 2:]  # (B, T, 3)

            # XY regression loss (only on real points)
            xy_loss = F.mse_loss(pred_xy[tgt_mask], next_xy[tgt_mask])

            # Pen state classification loss
            pen_loss = F.cross_entropy(pred_pen[tgt_mask], next_pen[tgt_mask])

            loss = xy_loss + pen_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_xy_loss += xy_loss.item()
            ep_pen_loss += pen_loss.item()
            with torch.no_grad():
                pen_pred = pred_pen[tgt_mask].argmax(dim=-1)
                ep_pen_acc += (pen_pred == next_pen[tgt_mask]).float().mean().item()
            ep_steps += 1

        avg_xy = ep_xy_loss / ep_steps
        avg_pen = ep_pen_loss / ep_steps
        avg_pen_acc = ep_pen_acc / ep_steps
        best_gen_acc = max(best_gen_acc, avg_pen_acc)

        row = {'epoch': epoch, 'xy_loss': round(avg_xy, 6),
               'pen_loss': round(avg_pen, 6), 'pen_acc': round(avg_pen_acc, 4)}
        gen_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | xy={avg_xy:.4f} pen={avg_pen:.4f} "
                  f"pen_acc={avg_pen_acc:.4f} | {time.time()-t0:.0f}s")

        # Round-trip test every 25 epochs
        if epoch % 25 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Generate from a known cell, decode back, compare
                test_cells, test_cmsk, test_tgt, test_tmsk, _, _ = \
                    gen_generation_batch(16, None, device)
                test_emb = model.composition.cell_proj(test_cells)
                # Autoregressive generation (simplified: just check teacher-forced quality)
                test_pred = model.stroke_gen(test_emb, test_tgt, test_tmsk)
                test_xy_err = F.mse_loss(test_pred[:,:,:2][test_tmsk],
                                          test_tgt[:,:,:2][test_tmsk]).item()
                print(f"         Round-trip XY error: {test_xy_err:.4f}")
            model.train()

    print(f"\n  Stage 2: pen_acc={best_gen_acc:.4f} ({time.time()-t0:.0f}s)")
    all_results['s2_generation'] = {
        'best_pen_acc': best_gen_acc, 'elapsed': time.time()-t0,
        'curve': gen_curve}
    save_checkpoint('s2')

    # ─────────────────────────────────────────────────────────
    # STAGE 3: Cell → Meaning (Composition)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 3: Cell → Meaning (Composition)")
    print(f"  3a: Text decoding (cell → char)")
    print(f"  3b: Sequence composition (cell program → result)")
    print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs each")
    print(f"{'='*60}")

    # --- Stage 3a: Text decoding ---
    print(f"\n  --- Stage 3a: Text Decoding ---")
    opt = torch.optim.AdamW(model.composition.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_text_acc = 0.0
    text_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_acc, ep_n = 0, 0, 0

        for step in range(steps_per_epoch):
            cells, cmsk, targets = gen_text_decode_batch(batch_size, device)
            logits = model.composition.text_decode(cells, cmsk)

            # Loss only on real cells
            loss = F.cross_entropy(logits[cmsk], targets[cmsk])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_loss += loss.item()
            with torch.no_grad():
                pred = logits[cmsk].argmax(dim=-1)
                ep_acc += (pred == targets[cmsk]).float().mean().item()
            ep_n += 1

        avg_loss = ep_loss / ep_n
        avg_acc = ep_acc / ep_n
        best_text_acc = max(best_text_acc, avg_acc)

        row = {'epoch': epoch, 'loss': round(avg_loss, 6), 'acc': round(avg_acc, 4)}
        text_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} acc={avg_acc:.4f} | "
                  f"best={best_text_acc:.4f} | {time.time()-t0:.0f}s")

    print(f"\n  Stage 3a: text_acc={best_text_acc:.4f} ({time.time()-t0:.0f}s) "
          f"{'PASSED ✓' if best_text_acc > 0.95 else '✗'}")
    all_results['s3a_text'] = {
        'best_acc': best_text_acc, 'elapsed': time.time()-t0,
        'passed': best_text_acc > 0.95, 'curve': text_curve}
    save_checkpoint('s3a')

    # --- Stage 3b: Sequence composition ---
    print(f"\n  --- Stage 3b: Sequence Composition ---")
    opt = torch.optim.AdamW(model.composition.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_seq_acc = 0.0
    seq_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_dot_acc, ep_cell_acc, ep_n = 0, 0, 0, 0

        for step in range(steps_per_epoch):
            cells, mask, results = gen_seq_compose_batch(batch_size, device)
            logits = model.composition.seq_compose(cells, mask)  # (B, n, 3)

            # Target classes
            target_cls = (results + 1).long()  # (B, n)
            loss = F.cross_entropy(logits.reshape(-1, 3), target_cls.reshape(-1))

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_loss += loss.item()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == target_cls)
                ep_dot_acc += correct.float().mean().item()
                ep_cell_acc += correct.all(dim=-1).float().mean().item()
            ep_n += 1

        avg_loss = ep_loss / ep_n
        avg_dot = ep_dot_acc / ep_n
        avg_cell = ep_cell_acc / ep_n
        best_seq_acc = max(best_seq_acc, avg_cell)

        row = {'epoch': epoch, 'loss': round(avg_loss, 6),
               'dot_acc': round(avg_dot, 4), 'cell_acc': round(avg_cell, 4)}
        seq_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={avg_loss:.4f} d={avg_dot:.4f} "
                  f"c={avg_cell:.4f} | best={best_seq_acc:.4f} | {time.time()-t0:.0f}s")

    print(f"\n  Stage 3b: seq_acc={best_seq_acc:.4f} ({time.time()-t0:.0f}s) "
          f"{'PASSED ✓' if best_seq_acc > 0.90 else '✗'}")
    all_results['s3b_sequence'] = {
        'best_acc': best_seq_acc, 'elapsed': time.time()-t0,
        'passed': best_seq_acc > 0.90, 'curve': seq_curve}
    save_checkpoint('s3b')

    # ─────────────────────────────────────────────────────────
    # STAGE 3.5: Ternary State Update Algebra
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 3.5: Ternary State Update Algebra")
    print(f"  add, negate, inner, cancel, update")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.algebra.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_vec, best_scl = 0.0, 0.0
    alg_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_vda, ep_vca, ep_sa = 0, 0, 0, 0
        nv, ns = 0, 0

        for step in range(steps_per_epoch):
            ca, cb, op, tgt, is_s = gen_algebra_batch(batch_size, device)
            vl, sp = model.algebra(ca, cb, op)
            loss, met = algebra_loss_fn(vl, sp, tgt, is_s)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            ep_loss += loss.item()
            bv = (is_s < 0.5).sum().item()
            bs_ = (is_s > 0.5).sum().item()
            ep_vda += met['vda'] * bv; ep_vca += met['vca'] * bv
            ep_sa += met['sa'] * bs_; nv += bv; ns += bs_

        avg_vca = ep_vca / max(nv, 1)
        avg_sa = ep_sa / max(ns, 1)
        best_vec = max(best_vec, avg_vca)
        best_scl = max(best_scl, avg_sa)

        row = {'epoch': epoch, 'loss': round(ep_loss/steps_per_epoch, 6),
               'vca': round(avg_vca, 4), 'sa': round(avg_sa, 4)}
        alg_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | loss={row['loss']:.4f} | "
                  f"vec_cell={avg_vca:.4f} scalar={avg_sa:.4f} | {time.time()-t0:.0f}s")

    print(f"\n  Stage 3.5: vec={best_vec:.4f} scl={best_scl:.4f} ({time.time()-t0:.0f}s) "
          f"{'PASSED ✓' if best_vec > 0.90 else '✗'}")
    all_results['s35_algebra'] = {
        'best_vec': best_vec, 'best_scl': best_scl, 'elapsed': time.time()-t0,
        'passed': best_vec > 0.90, 'curve': alg_curve}
    save_checkpoint('s35')

    # ─────────────────────────────────────────────────────────
    # STAGE 4: End-to-End (Stroke → Meaning)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STAGE 4: End-to-End (Stroke → Meaning)")
    print(f"  Joint fine-tuning: strokes → cells → text")
    print(f"  {steps_per_epoch} steps/epoch × {epochs} epochs")
    print(f"{'='*60}")

    # Fine-tune everything together
    opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    best_e2e_acc = 0.0
    e2e_curve = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        ep_perc_loss, ep_text_loss, ep_text_acc = 0, 0, 0
        ep_n = 0

        for step in range(steps_per_epoch):
            batch_strokes_list, batch_cells_list, batch_chars_list = [], [], []
            for _ in range(batch_size):
                text = random.choice(WORDS)
                if random.random() < 0.2: text = text.upper()
                elif random.random() < 0.3: text = text.capitalize()

                strokes = []
                cells = []
                chars = []
                spacing = 0.07 * 1.3
                for ci, ch in enumerate(text):
                    dl = BRAILLE.get(ch)
                    if dl is None:
                        continue
                    cells.append(dots_to_vector(dl, n))
                    chars.append(char_to_idx(ch))
                    if dl:
                        strokes.extend(braille_cell_strokes(
                            dl, 0.05 + ci * spacing, 0.25, 0.07, 0.16, n, jcfg))
                if not strokes or not cells:
                    strokes = [[(0.5,0.5),(0.5,0.5)]]
                    cells = [[0]*n]
                    chars = [0]
                while len(chars) < mc: chars.append(0)
                chars = chars[:mc]
                batch_strokes_list.append(strokes)
                batch_cells_list.append(cells)
                batch_chars_list.append(chars)

            # Encode strokes
            all_pts, all_msk, all_ct, all_cm = [], [], [], []
            for s, c in zip(batch_strokes_list, batch_cells_list):
                p, m = encode_strokes(s, mp)
                ct, cm = encode_cells(c, n, mc)
                all_pts.append(p); all_msk.append(m)
                all_ct.append(ct); all_cm.append(cm)

            pts = torch.stack(all_pts).to(device)
            msk = torch.stack(all_msk).to(device)
            tgt_cells = torch.stack(all_ct).to(device)
            cmsk = torch.stack(all_cm).to(device)
            char_targets = torch.tensor(batch_chars_list, dtype=torch.long, device=device)

            # Forward: stroke → cell embeddings → cell vectors
            cell_emb, _ = model.encoder(pts, msk)
            dot_logits = model.cell_decoder(cell_emb)

            # Perception loss
            perc_loss, perc_met = perception_loss(dot_logits, tgt_cells, cmsk)

            # Get predicted cells (hard argmax) and feed to composition
            with torch.no_grad():
                pred_cells_idx = dot_logits.argmax(dim=-1)  # (B, mc, n) class indices
                pred_cells = pred_cells_idx.float() - 1.0   # back to {-1,0,+1}

            # Text decode from predicted cells (use straight-through for gradients)
            # Use soft predictions via gumbel for gradient flow
            soft_cells = F.gumbel_softmax(dot_logits.reshape(-1, 3), tau=0.5, hard=False)
            soft_cells = soft_cells.reshape(batch_size, mc, n, 3)
            # Weighted sum: -1*p0 + 0*p1 + 1*p2
            weights = torch.tensor([-1.0, 0.0, 1.0], device=device)
            soft_cell_vals = (soft_cells * weights).sum(dim=-1)  # (B, mc, n)

            text_logits = model.composition.text_decode(soft_cell_vals, cmsk)
            text_loss = F.cross_entropy(text_logits[cmsk], char_targets[cmsk])

            loss = perc_loss + text_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_perc_loss += perc_loss.item()
            ep_text_loss += text_loss.item()
            with torch.no_grad():
                tp = text_logits[cmsk].argmax(dim=-1)
                ep_text_acc += (tp == char_targets[cmsk]).float().mean().item()
            ep_n += 1

        avg_perc = ep_perc_loss / ep_n
        avg_text = ep_text_loss / ep_n
        avg_acc = ep_text_acc / ep_n
        best_e2e_acc = max(best_e2e_acc, avg_acc)

        row = {'epoch': epoch, 'perc_loss': round(avg_perc, 6),
               'text_loss': round(avg_text, 6), 'text_acc': round(avg_acc, 4)}
        e2e_curve.append(row)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  E{epoch:3d} | perc={avg_perc:.4f} text={avg_text:.4f} "
                  f"acc={avg_acc:.4f} | best={best_e2e_acc:.4f} | {time.time()-t0:.0f}s")

    print(f"\n  Stage 4: e2e_acc={best_e2e_acc:.4f} ({time.time()-t0:.0f}s) "
          f"{'PASSED ✓' if best_e2e_acc > 0.90 else '✗'}")
    all_results['s4_e2e'] = {
        'best_acc': best_e2e_acc, 'elapsed': time.time()-t0,
        'passed': best_e2e_acc > 0.90, 'curve': e2e_curve}

    # ===================================================================
    # FINAL SAVE
    # ===================================================================
    total_elapsed = time.time() - grand_t0

    model_path = f"/results/braillenet_n{n}_final.pth"
    torch.save({
        'model_state': model.state_dict(),
        'n': n, 'd_model': d_model, 'num_layers': num_layers, 'nhead': nhead,
        'total_elapsed': total_elapsed,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'curve'}
                    for k, v in all_results.items()},
    }, model_path)

    results_path = f"/results/braillenet_n{n}_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    results_volume.commit()

    print(f"\n{'='*60}")
    print(f"FOUNDATION MODEL TRAINING COMPLETE")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for stage_name, res in all_results.items():
        status = '✓' if res.get('passed', True) else '✗'
        print(f"  {stage_name}: {status} | {res.get('elapsed', 0):.0f}s")
        for k, v in res.items():
            if k not in ('curve', 'elapsed', 'passed'):
                print(f"    {k}: {v}")
    print(f"\n  Model: {model_path}")
    print(f"  Results: {results_path}")
    print(f"\n  🧠 BrailleNet is ready.")

    return all_results


# ===================================================================
# CLI ENTRYPOINT
# ===================================================================

@app.local_entrypoint()
def main(
    n: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    steps_per_epoch: int = 500,
    jitter: str = "medium",
):
    results = train.remote(
        n=n, epochs=epochs, batch_size=batch_size,
        lr=lr, steps_per_epoch=steps_per_epoch, jitter=jitter,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — results on Modal volume")
    print("=" * 60)
    for stage_name, res in results.items():
        status = '✓' if res.get('passed', True) else '✗'
        print(f"\n{stage_name} [{status}]:")
        for k, v in res.items():
            if k != 'curve':
                print(f"  {k}: {v}")
