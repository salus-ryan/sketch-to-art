"""
Braille data generator — Python port of draw-engine.js braille logic.

Generates (stroke_sequence, cell_vectors, text) triples for training
the braille foundation model. Operates entirely in stroke/cell space —
no images.

Supports:
  - n-dot braille (4–16)
  - Signed cells: b ∈ {-1, 0, +1}^n
  - Nemeth math tokens
  - Jitter augmentation for robustness

Output format:
  {
    "strokes": [[[x, y], ...], ...],     # polyline strokes, normalized [0,1]
    "cells": [[1, 0, -1, ...], ...],      # one signed vector per cell
    "text": "hello",                       # original text
    "n": 8,
    "signed": true
  }
"""

import json
import math
import random
from itertools import product as cartesian_product
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# BRAILLE LOOKUP TABLE (8-dot, same as draw-engine.js)
# ═══════════════════════════════════════════════════════════════════

BRAILLE = {
    # Lowercase — base 6-dot patterns, NO dot 7 or 8
    'a': [1],             'b': [1,2],           'c': [1,4],           'd': [1,4,5],
    'e': [1,5],           'f': [1,2,4],         'g': [1,2,4,5],       'h': [1,2,5],
    'i': [2,4],           'j': [2,4,5],         'k': [1,3],           'l': [1,2,3],
    'm': [1,3,4],         'n': [1,3,4,5],       'o': [1,3,5],         'p': [1,2,3,4],
    'q': [1,2,3,4,5],     'r': [1,2,3,5],       's': [2,3,4],         't': [2,3,4,5],
    'u': [1,3,6],         'v': [1,2,3,6],       'w': [2,4,5,6],       'x': [1,3,4,6],
    'y': [1,3,4,5,6],     'z': [1,3,5,6],
    # Uppercase — same patterns + dot 7 (capital indicator)
    'A': [1,7],           'B': [1,2,7],         'C': [1,4,7],         'D': [1,4,5,7],
    'E': [1,5,7],         'F': [1,2,4,7],       'G': [1,2,4,5,7],     'H': [1,2,5,7],
    'I': [2,4,7],         'J': [2,4,5,7],       'K': [1,3,7],         'L': [1,2,3,7],
    'M': [1,3,4,7],       'N': [1,3,4,5,7],     'O': [1,3,5,7],       'P': [1,2,3,4,7],
    'Q': [1,2,3,4,5,7],   'R': [1,2,3,5,7],     'S': [2,3,4,7],       'T': [2,3,4,5,7],
    'U': [1,3,6,7],       'V': [1,2,3,6,7],     'W': [2,4,5,6,7],     'X': [1,3,4,6,7],
    'Y': [1,3,4,5,6,7],   'Z': [1,3,5,6,7],
    # Digits — base pattern + dot 8 (number indicator)
    '0': [2,4,5,8],       '1': [1,8],            '2': [1,2,8],          '3': [1,4,8],
    '4': [1,4,5,8],       '5': [1,5,8],          '6': [1,2,4,8],        '7': [1,2,4,5,8],
    '8': [1,2,5,8],       '9': [2,4,8],
    # Punctuation
    ' ': [],
    '.': [2,5,6],
    ',': [2],
    '!': [2,3,5],
    '?': [2,3,6],
    ':': [2,5],
    '-': [3,6],
    '#': [3,4,5,6],
    # Operators
    '+': [3,4,6],
    '=': [4,6],
    '*': [1,6],
    '/': [3,4],
    '(': [1,2,3,5,6],
    ')': [2,3,4,5,6],
    # Nemeth tokens (§-prefixed)
    '§int': [2,3,4,6],
    '§sum': [1,4,6],
    '§inf': [1,2,3,4,5,6],
    '§pi': [1,2,4,6],
    '§sqrt': [3,4,5],
    '§exp': [4,5],
    '§sub': [5,6],
    '§frac': [1,4,5,6],
    '§endf': [3,4,5,6],
    '§dx': [1,4,5],
    '§lim': [1,2,3],
    '§arr': [2,5,6],
    '§theta': [1,4,5,6],
    '§alpha': [1],
    '§beta': [1,2],
    '§gamma': [1,2,4,5],
    '§delta': [1,4,5],
    '§sigma': [2,3,4],
    '§omega': [2,4,5,6],
}


# ═══════════════════════════════════════════════════════════════════
# DOT POSITIONS — generalized n-dot grid
# ═══════════════════════════════════════════════════════════════════

def dot_positions(n):
    """Compute dot positions for n-dot braille, normalized to [0,1] within cell.

    Layout: 2 columns, ⌈n/2⌉ rows. Column-major numbering:
      [1]   [k+1]
      [2]   [k+2]
      ...   [...]
      [k]   [2k]     where k = ⌈n/2⌉
    """
    rows = math.ceil(n / 2)
    pos = {}
    for i in range(1, n + 1):
        col = 0 if i <= rows else 1
        row = (i - 1) if i <= rows else (i - rows - 1)
        pos[i] = (
            0.3 if col == 0 else 0.7,
            (row + 0.5) / rows,
        )
    return pos

# Cache for common n values
_DOT_POS_CACHE = {}
def get_dot_positions(n):
    if n not in _DOT_POS_CACHE:
        _DOT_POS_CACHE[n] = dot_positions(n)
    return _DOT_POS_CACHE[n]


# ═══════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════

def tokenize_braille(text):
    """Tokenize a string into braille tokens — handles §-prefixed Nemeth symbols."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i] == '§':
            end = i + 1
            while end < len(text) and text[end] != ' ' and text[end] != '§' \
                    and text[end] not in BRAILLE:
                end += 1
            tokens.append(text[i:end])
            i = end
        else:
            tokens.append(text[i])
            i += 1
    return tokens


# ═══════════════════════════════════════════════════════════════════
# STROKE GENERATION — pure geometry, no rendering
# ═══════════════════════════════════════════════════════════════════

def braille_cell_strokes(dots, ox, oy, cw, ch, n=8, jitter=None):
    """Generate strokes for a single n-dot braille cell.

    Args:
        dots: list of dot numbers [1,3,5] (all positive = asserted)
              or dict {dot: value} for signed braille (value in {-1, 0, +1})
        ox, oy: cell origin (top-left), normalized coordinates
        cw, ch: cell width/height
        n: number of dots in the braille system
        jitter: None or dict with keys:
            'position': float, max displacement as fraction of cell size
            'scale': float, scale variation (e.g., 0.1 = ±10%)
            'rotation': float, rotation in radians

    Returns:
        list of strokes, each a list of (x, y) tuples
    """
    strokes = []
    positions = get_dot_positions(n)
    dot_r = min(cw, ch) * (0.6 / math.ceil(n / 2))
    segments = 8

    # Normalize input
    if isinstance(dots, list):
        signed_dots = {abs(d): (1 if d > 0 else -1) for d in dots}
    else:
        signed_dots = dict(dots)

    # Apply jitter to cell
    if jitter:
        pos_noise = jitter.get('position', 0)
        scale_var = jitter.get('scale', 0)
        rot = jitter.get('rotation', 0)

        ox += random.uniform(-pos_noise, pos_noise) * cw
        oy += random.uniform(-pos_noise, pos_noise) * ch
        s = 1.0 + random.uniform(-scale_var, scale_var)
        cw *= s
        ch *= s
        dot_r *= s
        theta = random.uniform(-rot, rot)
    else:
        theta = 0

    for d_str, value in signed_dots.items():
        d = int(d_str) if isinstance(d_str, str) else d_str
        if d not in positions or value == 0:
            continue
        dx, dy = positions[d]
        cx = ox + dx * cw
        cy = oy + dy * ch

        # Apply rotation around cell center
        if theta != 0:
            cell_cx = ox + 0.5 * cw
            cell_cy = oy + 0.5 * ch
            rx = cx - cell_cx
            ry = cy - cell_cy
            cx = cell_cx + rx * math.cos(theta) - ry * math.sin(theta)
            cy = cell_cy + rx * math.sin(theta) + ry * math.cos(theta)

        if value > 0:
            # Asserted: filled circle
            pts = []
            for i in range(segments + 1):
                a = (i / segments) * math.pi * 2
                px = cx + math.cos(a) * dot_r
                py = cy + math.sin(a) * dot_r
                if jitter:
                    px += random.gauss(0, jitter.get('position', 0) * dot_r * 0.3)
                    py += random.gauss(0, jitter.get('position', 0) * dot_r * 0.3)
                pts.append((px, py))
            strokes.append(pts)
        else:
            # Denied/inhibited: X mark
            r = dot_r * 0.9
            strokes.append([(cx - r, cy - r), (cx + r, cy + r)])
            strokes.append([(cx + r, cy - r), (cx - r, cy + r)])

    return strokes


def braille_text_strokes(text, start_x=0.05, start_y=0.25, cell_w=0.07, cell_h=0.16,
                         n=8, jitter=None):
    """Generate strokes for a text string rendered as braille.

    Returns:
        tuple: (strokes, cells, tokens)
            strokes: list of polyline strokes
            cells: list of cell vectors (each a list of n values in {-1, 0, +1})
            tokens: list of text tokens
    """
    strokes = []
    cells = []
    valid_tokens = []
    spacing = cell_w * 1.3
    tokens = tokenize_braille(text)

    for i, token in enumerate(tokens):
        dot_list = BRAILLE.get(token)
        if dot_list is None:
            continue

        ox = start_x + i * spacing
        oy = start_y

        # Convert dot list to cell vector
        cell_vec = dots_to_vector(dot_list, n)
        cells.append(cell_vec)
        valid_tokens.append(token)

        if len(dot_list) == 0:
            continue  # space — no strokes but cell is all zeros

        cell_strokes = braille_cell_strokes(dot_list, ox, oy, cell_w, cell_h, n=n, jitter=jitter)
        strokes.extend(cell_strokes)

    return strokes, cells, valid_tokens


# ═══════════════════════════════════════════════════════════════════
# CELL VECTOR CONVERSION
# ═══════════════════════════════════════════════════════════════════

def dots_to_vector(dot_list, n=8):
    """Convert a list of dot numbers to an n-dimensional signed vector.

    [1, 3, 5] with n=8 → [1, 0, 1, 0, 1, 0, 0, 0]
    """
    vec = [0] * n
    for d in dot_list:
        idx = abs(d) - 1  # 1-indexed → 0-indexed
        if 0 <= idx < n:
            vec[idx] = 1 if d > 0 else -1
    return vec


def vector_to_dots(vec):
    """Convert an n-dimensional vector to a dot list.

    [1, 0, -1, 0, 1, 0, 0, 0] → {1: 1, 3: -1, 5: 1}
    """
    dots = {}
    for i, v in enumerate(vec):
        if v != 0:
            dots[i + 1] = int(v)
    return dots


# ═══════════════════════════════════════════════════════════════════
# RANDOM CELL GENERATION
# ═══════════════════════════════════════════════════════════════════

def random_cell(n=8, signed=False, density=0.5):
    """Generate a random cell vector.

    Args:
        n: number of dots
        signed: if True, values in {-1, 0, +1}; otherwise {0, 1}
        density: probability of each dot being non-zero
    """
    if signed:
        return [random.choice([-1, 0, 1]) if random.random() < density else 0
                for _ in range(n)]
    else:
        return [1 if random.random() < density else 0 for _ in range(n)]


def random_signed_cell(n=8):
    """Generate a uniformly random signed cell: each dot independently in {-1, 0, +1}."""
    return [random.choice([-1, 0, 1]) for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════
# ALGEBRAIC OPERATIONS (for Stage 3.5: ternary update algebra)
# ═══════════════════════════════════════════════════════════════════

def cell_add(a, b):
    """Clipped addition: clip(a + b, -1, +1)."""
    return [max(-1, min(1, ai + bi)) for ai, bi in zip(a, b)]


def cell_negate(a):
    """Negation: -a."""
    return [-v for v in a]


def cell_inner(a, b):
    """Inner product: ⟨a, b⟩."""
    return sum(ai * bi for ai, bi in zip(a, b))


def cell_cancel(a):
    """a + (-a) = 0. Returns the zero vector."""
    return [0] * len(a)


def cell_update(memory, delta):
    """Ternary state update: m'ᵢ = clip(mᵢ + Δᵢ, -1, +1).

    This is the fundamental update operation in the ternary algebra.
    The cancellation property (+1)+(-1)=0 is a native operation.
    """
    return cell_add(memory, delta)


# ═══════════════════════════════════════════════════════════════════
# SAMPLE GENERATION
# ═══════════════════════════════════════════════════════════════════

# Word lists for training data
WORDS = [
    'hello', 'world', 'braille', 'dots', 'read', 'write', 'learn',
    'touch', 'feel', 'see', 'art', 'draw', 'sketch', 'line', 'curve',
    'math', 'code', 'data', 'text', 'sign', 'type', 'hand',
    'cat', 'dog', 'sun', 'moon', 'star', 'tree', 'home', 'book',
    'pen', 'ink', 'page', 'word', 'cell', 'grid', 'dot', 'bump',
]

MATH_EXPRESSIONS = [
    'a+b=c',
    '1+1=2',
    'x§exp2',
    '§pi',
    'e§exp(§pi)',
    '§int§dx',
    '§sum',
    'a§exp2+b§exp2=c§exp2',
]

def generate_text_sample(n=8, signed=False, jitter=None, mode='word'):
    """Generate a single (strokes, cells, text) training sample.

    Args:
        n: number of dots in braille system
        signed: if True, use signed braille (not applicable for text lookup)
        jitter: jitter configuration dict or None
        mode: 'word', 'letter', 'number', 'math', or 'mixed'

    Returns:
        dict with keys: strokes, cells, text, n, signed
    """
    if mode == 'letter':
        text = random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    elif mode == 'number':
        text = str(random.randint(0, 9999))
    elif mode == 'math':
        text = random.choice(MATH_EXPRESSIONS)
    elif mode == 'mixed':
        mode = random.choice(['word', 'letter', 'number', 'math'])
        return generate_text_sample(n=n, signed=signed, jitter=jitter, mode=mode)
    else:  # word
        text = random.choice(WORDS)
        if random.random() < 0.2:
            text = text.upper()
        elif random.random() < 0.3:
            text = text.capitalize()

    strokes, cells, tokens = braille_text_strokes(text, n=n, jitter=jitter)

    return {
        'strokes': [[(round(x, 5), round(y, 5)) for x, y in s] for s in strokes],
        'cells': cells,
        'text': text,
        'tokens': tokens,
        'n': n,
        'signed': signed,
    }


def generate_raw_cell_sample(n=8, signed=True, jitter=None):
    """Generate a sample from a random cell vector (not from the lookup table).

    This is for training on the full {-1,0,+1}^n space, not just
    the ~100 characters in the BRAILLE table.

    Returns:
        dict with keys: strokes, cells, n, signed
    """
    if signed:
        cell_vec = random_signed_cell(n)
    else:
        cell_vec = random_cell(n, signed=False, density=0.5)

    dots = vector_to_dots(cell_vec)

    # Place cell at a random position
    ox = random.uniform(0.1, 0.6)
    oy = random.uniform(0.1, 0.6)
    cw = random.uniform(0.08, 0.15)
    ch = random.uniform(0.12, 0.25)

    strokes = braille_cell_strokes(dots, ox, oy, cw, ch, n=n, jitter=jitter)

    return {
        'strokes': [[(round(x, 5), round(y, 5)) for x, y in s] for s in strokes],
        'cells': [cell_vec],
        'cell_origin': (round(ox, 5), round(oy, 5)),
        'cell_size': (round(cw, 5), round(ch, 5)),
        'n': n,
        'signed': signed,
    }


def generate_algebra_sample(n=8):
    """Generate a sample for algebraic reasoning training.

    Returns a sample with two cells, an operation, and the result.
    """
    a = random_signed_cell(n)
    b = random_signed_cell(n)
    op = random.choice(['add', 'negate', 'inner', 'cancel', 'update'])

    if op == 'add':
        result = cell_add(a, b)
        result_type = 'vector'
    elif op == 'negate':
        result = cell_negate(a)
        result_type = 'vector'
    elif op == 'inner':
        ip = cell_inner(a, b)
        result = ip
        result_type = 'scalar'
    elif op == 'cancel':
        b = cell_negate(a)
        result = cell_cancel(a)
        result_type = 'vector'
    elif op == 'update':
        result = cell_update(a, b)  # a is memory, b is delta
        result_type = 'vector'

    return {
        'cell_a': a,
        'cell_b': b,
        'operation': op,
        'result': result,
        'result_type': result_type,
        'n': n,
    }


def generate_sequence_sample(n=8, signed=True, max_len=8):
    """Generate a stack program sample for composition training.

    Same semantics as braille_composition_test.py Task 3.
    """
    alphabet = [-1, 0, 1] if signed else [0, 1]
    seq_len = random.randint(3, max_len)
    cells = []
    stack = []

    # Instruction cells (special patterns)
    rows = math.ceil(n / 2)
    PUSH = tuple([1] + [0] * (n - 1))
    ADD = tuple([0, 1] + [0] * (n - 2))
    NEG = tuple([0, 0, 1] + [0] * (n - 3)) if n >= 3 else None
    DOT = tuple([1, 1] + [0] * (n - 2))

    for _ in range(seq_len):
        cell = tuple(random.choice(alphabet) for _ in range(n))
        cells.append(list(cell))

        if cell == ADD and len(stack) >= 2:
            a, b = stack.pop(), stack.pop()
            stack.append(cell_add(a, b))
        elif NEG and cell == NEG and len(stack) >= 1:
            stack.append(cell_negate(stack.pop()))
        elif cell == DOT and len(stack) >= 2:
            a, b = stack.pop(), stack.pop()
            ip = cell_inner(a, b) / n
            stack.append([round(ip, 6)] * n)
        else:
            stack.append(list(cell))

    if not stack:
        stack.append([0] * n)

    # Pad to max_len
    while len(cells) < max_len:
        cells.append([0] * n)

    return {
        'cells': cells[:max_len],
        'result': stack[-1],
        'n': n,
        'signed': signed,
    }


# ═══════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ═══════════════════════════════════════════════════════════════════

# Jitter presets
JITTER_NONE = None
JITTER_LIGHT = {'position': 0.02, 'scale': 0.05, 'rotation': 0.05}
JITTER_MEDIUM = {'position': 0.05, 'scale': 0.1, 'rotation': 0.1}
JITTER_HEAVY = {'position': 0.1, 'scale': 0.15, 'rotation': 0.2}


def generate_dataset(num_samples=10000, n=8, signed=False, jitter_preset='medium',
                     modes=None, include_raw_cells=True, include_algebra=True,
                     include_sequences=True, seed=42):
    """Generate a complete training dataset.

    Args:
        num_samples: total number of samples
        n: dot count
        signed: use ternary alphabet
        jitter_preset: 'none', 'light', 'medium', 'heavy'
        modes: list of text modes to include, default all
        include_raw_cells: include random cell samples
        include_algebra: include algebraic operation samples
        include_sequences: include stack program samples
        seed: random seed

    Returns:
        list of sample dicts
    """
    random.seed(seed)
    jitter = {
        'none': JITTER_NONE,
        'light': JITTER_LIGHT,
        'medium': JITTER_MEDIUM,
        'heavy': JITTER_HEAVY,
    }.get(jitter_preset, JITTER_MEDIUM)

    if modes is None:
        modes = ['word', 'letter', 'number', 'math', 'mixed']

    samples = []
    types = []

    # Determine sample distribution
    if include_raw_cells:
        types.append('raw_cell')
    if include_algebra:
        types.append('algebra')
    if include_sequences:
        types.append('sequence')
    types.append('text')  # always include text samples

    for i in range(num_samples):
        sample_type = types[i % len(types)]

        if sample_type == 'text':
            mode = modes[i % len(modes)]
            sample = generate_text_sample(n=n, signed=signed, jitter=jitter, mode=mode)
            sample['type'] = 'text'
        elif sample_type == 'raw_cell':
            sample = generate_raw_cell_sample(n=n, signed=signed or True, jitter=jitter)
            sample['type'] = 'raw_cell'
        elif sample_type == 'algebra':
            sample = generate_algebra_sample(n=n)
            sample['type'] = 'algebra'
        elif sample_type == 'sequence':
            sample = generate_sequence_sample(n=n, signed=signed or True)
            sample['type'] = 'sequence'

        samples.append(sample)

    return samples


def save_dataset(samples, path):
    """Save dataset as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


def load_dataset(path):
    """Load dataset from JSON lines."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate braille training data')
    parser.add_argument('--n', type=int, default=8, help='Number of dots')
    parser.add_argument('--signed', action='store_true', help='Use signed {-1,0,+1}')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--jitter', default='medium', choices=['none', 'light', 'medium', 'heavy'])
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--preview', type=int, default=3, help='Print N samples to stdout')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        sign_tag = 's' if args.signed else ''
        args.output = f'results/braille_data_n{args.n}{sign_tag}_{args.samples}.jsonl'

    print(f"Generating {args.samples} samples: n={args.n}, "
          f"signed={args.signed}, jitter={args.jitter}")

    samples = generate_dataset(
        num_samples=args.samples,
        n=args.n,
        signed=args.signed,
        jitter_preset=args.jitter,
        seed=args.seed,
    )

    # Preview
    for i, s in enumerate(samples[:args.preview]):
        print(f"\n--- Sample {i} ({s.get('type', '?')}) ---")
        if 'text' in s:
            print(f"  text: {s['text']}")
        if 'cells' in s:
            print(f"  cells ({len(s['cells'])}): {s['cells'][:3]}{'...' if len(s['cells']) > 3 else ''}")
        if 'strokes' in s:
            print(f"  strokes: {len(s['strokes'])} polylines")
        if 'operation' in s:
            print(f"  op: {s['operation']}, result_type: {s['result_type']}")
        if 'result' in s:
            r = s['result']
            print(f"  result: {r if isinstance(r, (int, float)) else r[:5]}")

    save_dataset(samples, args.output)

    # Stats
    type_counts = {}
    for s in samples:
        t = s.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nType distribution: {type_counts}")
