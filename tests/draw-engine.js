/**
 * Drawing engine for Playwright — teaches the AI by drawing.
 *
 * All coordinates are normalized 0–1. The engine maps them to the canvas
 * bounding box and drives Playwright mouse events.
 */

// ─── Stroke Font (A–Z, 0–9, punctuation) ───
const FONT = {
  a: [[[0,1],[0.3,0],[0.6,1]],[[0.12,0.6],[0.48,0.6]]],
  b: [[[0.1,0],[0.1,1]],[[0.1,0],[0.45,0],[0.5,0.15],[0.45,0.3],[0.1,0.35]],[[0.1,0.35],[0.5,0.4],[0.55,0.55],[0.55,0.7],[0.45,0.9],[0.1,1]]],
  c: [[[0.6,0.15],[0.4,0],[0.15,0.15],[0.1,0.5],[0.15,0.85],[0.4,1],[0.6,0.85]]],
  d: [[[0.1,0],[0.1,1]],[[0.1,0],[0.35,0],[0.55,0.15],[0.6,0.5],[0.55,0.85],[0.35,1],[0.1,1]]],
  e: [[[0.1,0],[0.1,1]],[[0.1,0],[0.6,0]],[[0.1,0.5],[0.5,0.5]],[[0.1,1],[0.6,1]]],
  f: [[[0.2,0],[0.2,1]],[[0,0.3],[0.6,0.3]],[[0,0],[0.6,0]]],
  g: [[[0.6,0.15],[0.4,0],[0.15,0.15],[0.1,0.5],[0.15,0.85],[0.4,1],[0.6,0.85],[0.6,0.5],[0.4,0.5]]],
  h: [[[0.1,0],[0.1,1]],[[0.5,0],[0.5,1]],[[0.1,0.5],[0.5,0.5]]],
  i: [[[0.3,0.2],[0.3,1]],[[0.3,0],[0.3,0.05]]],
  j: [[[0.5,0],[0.5,0.85],[0.35,1],[0.15,0.85]]],
  k: [[[0.1,0],[0.1,1]],[[0.5,0],[0.1,0.5],[0.5,1]]],
  l: [[[0.15,0],[0.15,1]],[[0.15,1],[0.55,1]]],
  m: [[[0.05,1],[0.05,0],[0.3,0.4],[0.55,0],[0.55,1]]],
  n: [[[0.1,1],[0.1,0]],[[0.1,0],[0.5,1]],[[0.5,1],[0.5,0]]],
  o: [[[0.3,0],[0.1,0.15],[0.05,0.5],[0.1,0.85],[0.3,1],[0.5,0.85],[0.55,0.5],[0.5,0.15],[0.3,0]]],
  p: [[[0.1,0],[0.1,1]],[[0.1,0],[0.45,0],[0.55,0.15],[0.55,0.3],[0.45,0.45],[0.1,0.5]]],
  q: [[[0.3,0],[0.1,0.15],[0.05,0.5],[0.1,0.85],[0.3,1],[0.5,0.85],[0.55,0.5],[0.5,0.15],[0.3,0]],[[0.4,0.8],[0.6,1.05]]],
  r: [[[0.1,0],[0.1,1]],[[0.1,0],[0.45,0],[0.55,0.15],[0.55,0.3],[0.45,0.45],[0.1,0.5]],[[0.35,0.45],[0.55,1]]],
  s: [[[0.55,0.1],[0.4,0],[0.15,0.05],[0.1,0.2],[0.15,0.4],[0.45,0.6],[0.5,0.8],[0.45,0.95],[0.2,1],[0.1,0.9]]],
  t: [[[0.3,0],[0.3,1]],[[0,0],[0.6,0]]],
  u: [[[0.1,0],[0.1,0.8],[0.2,1],[0.4,1],[0.5,0.8],[0.5,0]]],
  v: [[[0.05,0],[0.3,1],[0.55,0]]],
  w: [[[0.0,0],[0.15,1],[0.3,0.4],[0.45,1],[0.6,0]]],
  x: [[[0.1,0],[0.5,1]],[[0.5,0],[0.1,1]]],
  y: [[[0.1,0],[0.3,0.5]],[[0.5,0],[0.3,0.5],[0.2,1]]],
  z: [[[0.1,0],[0.5,0],[0.1,1],[0.5,1]]],
  ' ': [],
  ':': [[[0.3,0.2],[0.3,0.25]],[[0.3,0.75],[0.3,0.8]]],
  '.': [[[0.3,0.95],[0.3,1]]],
  '-': [[[0.1,0.5],[0.5,0.5]]],
  '!': [[[0.3,0],[0.3,0.7]],[[0.3,0.85],[0.3,0.9]]],
  '0': [[[0.3,0],[0.1,0.15],[0.05,0.5],[0.1,0.85],[0.3,1],[0.5,0.85],[0.55,0.5],[0.5,0.15],[0.3,0]],[[0.15,0.8],[0.45,0.2]]],
  '1': [[[0.2,0.15],[0.35,0]],[[0.35,0],[0.35,1]],[[0.15,1],[0.55,1]]],
  '2': [[[0.1,0.15],[0.25,0],[0.45,0],[0.55,0.15],[0.55,0.35],[0.1,1],[0.55,1]]],
  '3': [[[0.1,0.1],[0.3,0],[0.5,0.1],[0.5,0.3],[0.35,0.45]],[[0.35,0.45],[0.5,0.6],[0.5,0.85],[0.3,1],[0.1,0.9]]],
};

// ─── Braille (8-dot native) ───
// 8-dot computer braille: dots numbered 1-8 in a 2×4 grid
//   [1] [4]
//   [2] [5]
//   [3] [6]
//   [7] [8]
// 256 possible patterns (vs 64 for 6-dot).
// Dot 7 = capital indicator, dot 8 = number indicator.
const BRAILLE = {
  // Lowercase — base 6-dot patterns, NO dot 7 or 8
  a: [1],             b: [1,2],           c: [1,4],           d: [1,4,5],
  e: [1,5],           f: [1,2,4],         g: [1,2,4,5],       h: [1,2,5],
  i: [2,4],           j: [2,4,5],         k: [1,3],           l: [1,2,3],
  m: [1,3,4],         n: [1,3,4,5],       o: [1,3,5],         p: [1,2,3,4],
  q: [1,2,3,4,5],     r: [1,2,3,5],       s: [2,3,4],         t: [2,3,4,5],
  u: [1,3,6],         v: [1,2,3,6],       w: [2,4,5,6],       x: [1,3,4,6],
  y: [1,3,4,5,6],     z: [1,3,5,6],
  // Uppercase — same patterns + dot 7 (capital indicator)
  A: [1,7],           B: [1,2,7],         C: [1,4,7],         D: [1,4,5,7],
  E: [1,5,7],         F: [1,2,4,7],       G: [1,2,4,5,7],     H: [1,2,5,7],
  I: [2,4,7],         J: [2,4,5,7],       K: [1,3,7],         L: [1,2,3,7],
  M: [1,3,4,7],       N: [1,3,4,5,7],     O: [1,3,5,7],       P: [1,2,3,4,7],
  Q: [1,2,3,4,5,7],   R: [1,2,3,5,7],     S: [2,3,4,7],       T: [2,3,4,5,7],
  U: [1,3,6,7],       V: [1,2,3,6,7],     W: [2,4,5,6,7],     X: [1,3,4,6,7],
  Y: [1,3,4,5,6,7],   Z: [1,3,5,6,7],
  // Digits — base pattern + dot 8 (number indicator)
  '0': [2,4,5,8],     '1': [1,8],          '2': [1,2,8],        '3': [1,4,8],
  '4': [1,4,5,8],     '5': [1,5,8],        '6': [1,2,4,8],      '7': [1,2,4,5,8],
  '8': [1,2,5,8],     '9': [2,4,8],
  // Punctuation — 6-dot only (no modifiers)
  ' ': [],
  '.': [2,5,6],
  ',': [2],
  '!': [2,3,5],
  '?': [2,3,6],
  ':': [2,5],
  '-': [3,6],
  '#': [3,4,5,6],
  // ─── Nemeth Braille math symbols ───
  // Operators
  '+': [3,4,6],
  '=': [4,6],                 // Nemeth equals (simplified to one cell)
  '*': [1,6],                 // multiplication dot
  '/': [3,4],                 // fraction line
  '(': [1,2,3,5,6],
  ')': [2,3,4,5,6],
  // Nemeth-specific tokens (use § prefix for multi-cell symbols)
  '§int': [2,3,4,6],          // integral sign ∫
  '§sum': [1,4,6],            // summation Σ
  '§inf': [1,2,3,4,5,6],      // infinity ∞
  '§pi': [1,2,4,6],           // π
  '§sqrt': [3,4,5],           // square root prefix √
  '§exp': [4,5],              // superscript indicator (exponent)
  '§sub': [5,6],              // subscript indicator
  '§frac': [1,4,5,6],         // fraction open
  '§endf': [3,4,5,6],         // fraction close
  '§dx': [1,4,5],             // dx (differential)
  '§lim': [1,2,3],            // lim
  '§arr': [2,5,6],            // arrow →
  '§theta': [1,4,5,6],        // θ
  '§alpha': [1],              // α (same as 'a' in Nemeth context)
  '§beta': [1,2],             // β
  '§gamma': [1,2,4,5],        // γ
  '§delta': [1,4,5],          // δ
  '§sigma': [2,3,4],          // σ
  '§omega': [2,4,5,6],        // ω
};

// ─── Generalized n-dot Braille ───
// b ∈ {-1, 0, +1}^n
// Dots arranged in a 2×⌈n/2⌉ grid. Column-major numbering:
//   [1] [k+1]
//   [2] [k+2]
//   ... [...]
//   [k] [2k]     where k = ⌈n/2⌉

// Compute dot positions for n-dot braille (normalized 0–1 within cell)
function dotPositions(n) {
  const rows = Math.ceil(n / 2);
  const pos = {};
  for (let i = 1; i <= n; i++) {
    const col = i <= rows ? 0 : 1;              // left or right column
    const row = i <= rows ? i - 1 : i - rows - 1;
    pos[i] = [
      col === 0 ? 0.3 : 0.7,
      (row + 0.5) / rows,  // evenly spaced vertically
    ];
  }
  return pos;
}

// Precomputed positions for 8-dot (the standard)
const DOT_POS_8 = dotPositions(8);

// Draw a single n-dot braille cell at (ox, oy) with size (cw, ch)
// dots: array of dot numbers (positive = asserted) OR
//       object { dot: value } for signed braille (value in {-1, 0, +1})
function brailleCell(dots, ox, oy, cw, ch, n = 8) {
  const strokes = [];
  const positions = n === 8 ? DOT_POS_8 : dotPositions(n);
  const dotR = Math.min(cw, ch) * (0.6 / Math.ceil(n / 2)); // scale with n
  const segments = 8;

  // Normalize input: accept [1,3,5] or {1:+1, 3:-1, 5:+1}
  let signedDots;
  if (Array.isArray(dots)) {
    signedDots = {};
    for (const d of dots) signedDots[Math.abs(d)] = d > 0 ? 1 : -1;
  } else {
    signedDots = dots;
  }

  for (const [dStr, value] of Object.entries(signedDots)) {
    const d = parseInt(dStr);
    if (!positions[d] || value === 0) continue;
    const [dx, dy] = positions[d];
    const cx = ox + dx * cw;
    const cy = oy + dy * ch;

    if (value > 0) {
      // Asserted: filled circle
      const pts = [];
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2;
        pts.push([cx + Math.cos(a) * dotR, cy + Math.sin(a) * dotR]);
      }
      strokes.push(pts);
    } else {
      // Denied/inhibited: X mark
      const r = dotR * 0.9;
      strokes.push([[cx - r, cy - r], [cx + r, cy + r]]);
      strokes.push([[cx + r, cy - r], [cx - r, cy + r]]);
    }
  }
  return strokes;
}

// Draw text as braille cells
function brailleText(text, startX = 0.05, startY = 0.25, cellW = 0.07, cellH = 0.16) {
  const strokes = [];
  const spacing = cellW * 1.3;
  const tokens = tokenizeBraille(text);
  for (let i = 0; i < tokens.length; i++) {
    const dots = BRAILLE[tokens[i]];
    if (!dots || dots.length === 0) continue;  // space = no dots
    const ox = startX + i * spacing;
    const oy = startY;
    strokes.push(...brailleCell(dots, ox, oy, cellW, cellH));
  }
  return strokes;
}

// Tokenize a string into braille tokens — handles §-prefixed Nemeth symbols
function tokenizeBraille(text) {
  const tokens = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] === '§') {
      // Read until next space, §, or regular char boundary
      let end = i + 1;
      while (end < text.length && text[end] !== ' ' && text[end] !== '§'
             && !BRAILLE[text[end]]) {
        end++;
      }
      tokens.push(text.slice(i, end));
      i = end;
    } else {
      tokens.push(text[i]);
      i++;
    }
  }
  return tokens;
}

// ─── Shape Library ───
const SHAPES = {
  circle: (cx, cy, r, segments = 24) => {
    const pts = [];
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
    return [pts];
  },

  triangle: (cx, cy, r) => [[
    [cx, cy - r],
    [cx + r * 0.87, cy + r * 0.5],
    [cx - r * 0.87, cy + r * 0.5],
    [cx, cy - r],
  ]],

  square: (cx, cy, r) => [[
    [cx - r, cy - r], [cx + r, cy - r],
    [cx + r, cy + r], [cx - r, cy + r],
    [cx - r, cy - r],
  ]],

  star: (cx, cy, r) => {
    const pts = [];
    for (let i = 0; i <= 10; i++) {
      const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
      const sr = i % 2 === 0 ? r : r * 0.4;
      pts.push([cx + Math.cos(a) * sr, cy + Math.sin(a) * sr]);
    }
    return [pts];
  },

  heart: (cx, cy, r) => {
    const pts = [];
    for (let i = 0; i <= 30; i++) {
      const t = (i / 30) * Math.PI * 2;
      const x = cx + r * 0.5 * (16 * Math.sin(t) ** 3) / 16;
      const y = cy - r * 0.5 * (13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t)) / 16;
      pts.push([x, y]);
    }
    return [pts];
  },

  house: (cx, cy, r) => [
    // walls
    [[cx - r, cy], [cx - r, cy + r], [cx + r, cy + r], [cx + r, cy]],
    // roof
    [[cx - r * 1.1, cy], [cx, cy - r * 0.8], [cx + r * 1.1, cy]],
    // door
    [[cx - r * 0.2, cy + r], [cx - r * 0.2, cy + r * 0.4], [cx + r * 0.2, cy + r * 0.4], [cx + r * 0.2, cy + r]],
  ],

  cat: (cx, cy, r) => [
    // head circle
    ...SHAPES.circle(cx, cy, r * 0.4, 20),
    // left ear
    [[cx - r * 0.3, cy - r * 0.3], [cx - r * 0.4, cy - r * 0.6], [cx - r * 0.1, cy - r * 0.35]],
    // right ear
    [[cx + r * 0.3, cy - r * 0.3], [cx + r * 0.4, cy - r * 0.6], [cx + r * 0.1, cy - r * 0.35]],
    // left eye
    [[cx - r * 0.15, cy - r * 0.08], [cx - r * 0.12, cy - r * 0.12], [cx - r * 0.09, cy - r * 0.08]],
    // right eye
    [[cx + r * 0.09, cy - r * 0.08], [cx + r * 0.12, cy - r * 0.12], [cx + r * 0.15, cy - r * 0.08]],
    // nose
    [[cx, cy + r * 0.02], [cx - r * 0.04, cy + r * 0.07], [cx + r * 0.04, cy + r * 0.07], [cx, cy + r * 0.02]],
    // left whiskers
    [[cx - r * 0.4, cy + r * 0.0], [cx - r * 0.1, cy + r * 0.05]],
    [[cx - r * 0.4, cy + r * 0.1], [cx - r * 0.1, cy + r * 0.08]],
    // right whiskers
    [[cx + r * 0.1, cy + r * 0.05], [cx + r * 0.4, cy + r * 0.0]],
    [[cx + r * 0.1, cy + r * 0.08], [cx + r * 0.4, cy + r * 0.1]],
    // mouth
    [[cx - r * 0.06, cy + r * 0.12], [cx, cy + r * 0.16], [cx + r * 0.06, cy + r * 0.12]],
  ],

  tree: (cx, cy, r) => [
    // trunk
    [[cx - r * 0.1, cy + r], [cx - r * 0.1, cy - r * 0.2],],
    [[cx + r * 0.1, cy + r], [cx + r * 0.1, cy - r * 0.2],],
    // canopy
    ...SHAPES.triangle(cx, cy - r * 0.5, r * 0.5),
    ...SHAPES.triangle(cx, cy - r * 0.2, r * 0.6),
    ...SHAPES.triangle(cx, cy + r * 0.15, r * 0.7),
  ],

  braille_cell: (cx, cy, r) => {
    // Draw a random braille cell as a shape
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    const ch = letters[Math.floor(Math.random() * letters.length)];
    const dots = BRAILLE[ch] || [1];
    return brailleCell(dots, cx - r, cy - r * 1.2, r * 2, r * 2.4);
  },
};

// ─── Drawing Curriculum ───
// Each lesson generates training data for the AI
const CURRICULUM = [
  {
    name: 'braille-hello',
    description: 'Write "hello" in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('hello') }),
  },
  {
    name: 'braille-world',
    description: 'Write "world" in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('world') }),
  },
  {
    name: 'braille-alpha',
    description: 'Write a-m in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('abcdefghijklm', 0.02, 0.25, 0.055, 0.14) }),
  },
  {
    name: 'braille-alpha-2',
    description: 'Write n-z in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('nopqrstuvwxyz', 0.02, 0.25, 0.055, 0.14) }),
  },
  {
    name: 'commit-feat',
    description: 'Write "feat" in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('feat') }),
  },
  {
    name: 'commit-fix',
    description: 'Write "fix" in braille',
    draw: () => ({ type: 'raw', strokes: brailleText('fix') }),
  },
  // ─── Nemeth Math ───
  {
    name: 'math-integral',
    description: 'Solve ∫x²dx = x³/3 in Nemeth braille',
    draw: () => ({ type: 'multi', parts: [
      { type: 'raw', strokes: brailleText('§intx§exp2§dx', 0.05, 0.2, 0.06, 0.14) },
      { type: 'raw', strokes: brailleText('=', 0.05, 0.45, 0.06, 0.14) },
      { type: 'raw', strokes: brailleText('x§exp3/3', 0.14, 0.45, 0.06, 0.14) },
    ]}),
  },
  {
    name: 'math-euler',
    description: 'Write Euler\'s identity e^iπ + 1 = 0 in Nemeth braille',
    draw: () => ({ type: 'multi', parts: [
      { type: 'raw', strokes: brailleText('e§expi§pi', 0.05, 0.2, 0.065, 0.15) },
      { type: 'raw', strokes: brailleText('+1=0', 0.05, 0.5, 0.065, 0.15) },
    ]}),
  },
  {
    name: 'math-quadratic',
    description: 'Write the quadratic formula in Nemeth braille',
    draw: () => ({ type: 'multi', parts: [
      { type: 'raw', strokes: brailleText('x=', 0.03, 0.15, 0.05, 0.12) },
      { type: 'raw', strokes: brailleText('(-b+§sqrtb§exp2-4ac)', 0.03, 0.38, 0.04, 0.10) },
      { type: 'raw', strokes: brailleText('/2a', 0.03, 0.6, 0.05, 0.12) },
    ]}),
  },
  {
    name: 'math-pythagorean',
    description: 'Write a² + b² = c² in Nemeth braille',
    draw: () => ({ type: 'raw', strokes: brailleText('a§exp2+b§exp2=c§exp2', 0.05, 0.3, 0.06, 0.14) }),
  },
  {
    name: 'math-sum',
    description: 'Write Σ(n=1→∞) 1/n² = π²/6 in Nemeth braille',
    draw: () => ({ type: 'multi', parts: [
      { type: 'raw', strokes: brailleText('§sumn=1§arr§inf', 0.05, 0.15, 0.055, 0.13) },
      { type: 'raw', strokes: brailleText('1/n§exp2', 0.05, 0.4, 0.055, 0.13) },
      { type: 'raw', strokes: brailleText('=§pi§exp2/6', 0.05, 0.65, 0.055, 0.13) },
    ]}),
  },
  // ─── Generalized n-dot Braille ───
  {
    name: 'ndot-hierarchy',
    description: 'Draw B₆ ⊂ B₈ ⊂ B₁₀ — the braille hierarchy',
    draw: () => {
      const strokes = [];
      const labels = [6, 8, 10];
      const y = 0.15;
      for (let col = 0; col < 3; col++) {
        const n = labels[col];
        const ox = 0.05 + col * 0.32;
        // Draw a fully-asserted cell (all dots on) for each n
        const allDots = {};
        for (let d = 1; d <= n; d++) allDots[d] = 1;
        strokes.push(...brailleCell(allDots, ox, y, 0.12, 0.25, n));
        // Draw a half-asserted cell (odds only)
        const oddDots = {};
        for (let d = 1; d <= n; d += 2) oddDots[d] = 1;
        strokes.push(...brailleCell(oddDots, ox + 0.15, y, 0.12, 0.25, n));
      }
      // Label row: "6" "8" "10" in standard braille below
      strokes.push(...brailleText('6  8  10', 0.07, 0.55, 0.055, 0.13));
      return { type: 'raw', strokes };
    },
  },
  {
    name: 'ndot-negative',
    description: 'Negative-dot braille: b ∈ {-1, 0, +1}⁸ — assert, deny, absent',
    draw: () => {
      const strokes = [];
      // Cell 1: all asserted
      strokes.push(...brailleCell(
        {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1},
        0.05, 0.15, 0.14, 0.28, 8
      ));
      // Cell 2: mixed — some asserted, some denied
      strokes.push(...brailleCell(
        {1:1, 2:-1, 3:1, 4:-1, 5:1, 6:-1, 7:1, 8:-1},
        0.25, 0.15, 0.14, 0.28, 8
      ));
      // Cell 3: all denied
      strokes.push(...brailleCell(
        {1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:-1, 8:-1},
        0.45, 0.15, 0.14, 0.28, 8
      ));
      // Cell 4: sparse — only dots 1,4 asserted, 3,6 denied
      strokes.push(...brailleCell(
        {1:1, 3:-1, 4:1, 6:-1},
        0.65, 0.15, 0.14, 0.28, 8
      ));
      // Label
      strokes.push(...brailleText('+1 +- -1 ?', 0.07, 0.55, 0.055, 0.13));
      return { type: 'raw', strokes };
    },
  },
  {
    name: 'ndot-12',
    description: 'Draw 12-dot braille cells — 2×6 grid, 4096 patterns',
    draw: () => {
      const strokes = [];
      // Random 12-dot cells
      const patterns = [
        {1:1, 3:1, 5:1, 7:1, 9:1, 11:1},             // all left column
        {2:1, 4:1, 6:1, 8:1, 10:1, 12:1},             // all right column
        {1:1, 4:1, 5:1, 8:1, 9:1, 12:1},              // diagonal
        {1:1, 2:1, 3:-1, 4:-1, 7:1, 8:1, 11:-1, 12:-1}, // signed
        {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}, // full
      ];
      for (let i = 0; i < patterns.length; i++) {
        strokes.push(...brailleCell(patterns[i], 0.05 + i * 0.18, 0.1, 0.12, 0.4, 12));
      }
      return { type: 'raw', strokes };
    },
  },
  {
    name: 'ndot-cancellation',
    description: 'Demonstrate (+d) + (-d) = 0 — algebraic cancellation',
    draw: () => {
      const strokes = [];
      // Cell with +1 dots
      strokes.push(...brailleCell({1:1, 2:1, 3:1}, 0.05, 0.2, 0.12, 0.22, 6));
      // Plus sign
      strokes.push(...brailleText('+', 0.20, 0.25, 0.06, 0.14));
      // Cell with -1 dots (same positions)
      strokes.push(...brailleCell({1:-1, 2:-1, 3:-1}, 0.30, 0.2, 0.12, 0.22, 6));
      // Equals sign
      strokes.push(...brailleText('=', 0.45, 0.25, 0.06, 0.14));
      // Empty cell (cancelled out — draw the grid outline only)
      strokes.push(...brailleCell({}, 0.55, 0.2, 0.12, 0.22, 6));
      // Second row: full expression in text
      strokes.push(...brailleText('(+d)+(-d)=0', 0.05, 0.55, 0.05, 0.12));
      return { type: 'raw', strokes };
    },
  },
  {
    name: 'shapes',
    description: 'Draw basic geometric shapes',
    draw: () => ({ type: 'shapes', shapes: [
      { shape: 'circle', cx: 0.2, cy: 0.4, r: 0.12 },
      { shape: 'triangle', cx: 0.5, cy: 0.4, r: 0.12 },
      { shape: 'square', cx: 0.8, cy: 0.4, r: 0.1 },
    ]}),
  },
  {
    name: 'star-heart',
    description: 'Draw a star and a heart',
    draw: () => ({ type: 'shapes', shapes: [
      { shape: 'star', cx: 0.3, cy: 0.45, r: 0.18 },
      { shape: 'heart', cx: 0.7, cy: 0.45, r: 0.2 },
    ]}),
  },
  {
    name: 'house',
    description: 'Draw a simple house',
    draw: () => ({ type: 'shapes', shapes: [
      { shape: 'house', cx: 0.5, cy: 0.45, r: 0.2 },
    ]}),
  },
  {
    name: 'cat',
    description: 'Draw a cat face',
    draw: () => ({ type: 'shapes', shapes: [
      { shape: 'cat', cx: 0.5, cy: 0.45, r: 0.35 },
    ]}),
  },
  {
    name: 'scene',
    description: 'Draw a scene: house with tree',
    draw: () => ({ type: 'shapes', shapes: [
      { shape: 'house', cx: 0.35, cy: 0.5, r: 0.18 },
      { shape: 'tree', cx: 0.7, cy: 0.45, r: 0.2 },
    ]}),
  },
];

// ─── Drawing Engine ───
async function drawStrokes(page, canvasBox, strokes) {
  for (const stroke of strokes) {
    if (stroke.length < 2) continue;
    const toX = (nx) => canvasBox.x + nx * canvasBox.width;
    const toY = (ny) => canvasBox.y + ny * canvasBox.height;

    await page.mouse.move(toX(stroke[0][0]), toY(stroke[0][1]));
    await page.mouse.down();
    for (let pi = 1; pi < stroke.length; pi++) {
      await page.mouse.move(
        toX(stroke[pi][0]), toY(stroke[pi][1]), { steps: 2 }
      );
    }
    await page.mouse.up();
  }
}

async function drawText(page, canvasBox, text, opts = {}) {
  const letterW = 0.04;
  const letterH = 0.06;
  const spacing = letterW * 1.4;
  const startX = opts.startX || 0.1;
  const startY = opts.startY || 0.4;

  for (let ci = 0; ci < text.length; ci++) {
    const ch = text[ci].toLowerCase();
    const letterStrokes = FONT[ch] || [];
    const ox = startX + ci * spacing;
    const oy = startY;

    // Map letter-local [0–1] coords → canvas-normalized coords
    const mapped = letterStrokes.map(stroke =>
      stroke.map(([x, y]) => [ox + x * letterW, oy + y * letterH])
    );
    await drawStrokes(page, canvasBox, mapped);
  }
}

async function drawShape(page, canvasBox, shapeName, cx, cy, r) {
  const generator = SHAPES[shapeName];
  if (!generator) throw new Error(`Unknown shape: ${shapeName}`);
  const strokes = generator(cx, cy, r);
  await drawStrokes(page, canvasBox, strokes);
}

async function executePlan(page, canvasBox, plan) {
  if (plan.type === 'text') {
    await drawText(page, canvasBox, plan.text, {
      startX: plan.startX,
      startY: plan.startY,
    });
  } else if (plan.type === 'shapes') {
    for (const s of plan.shapes) {
      await drawShape(page, canvasBox, s.shape, s.cx, s.cy, s.r);
    }
  } else if (plan.type === 'raw') {
    await drawStrokes(page, canvasBox, plan.strokes);
  } else if (plan.type === 'multi') {
    for (const part of plan.parts) {
      await executePlan(page, canvasBox, part);
    }
  }
}

async function executeLesson(page, canvasBox, lesson) {
  await executePlan(page, canvasBox, lesson.draw());
}

// ─── Infinite Curriculum Kernel ───
// kernel(level, seed) → lesson
// Deterministic given (level, seed), but infinite in output space.
// Each level composes from the level below it.
//
// Level 0: Primitives    — single strokes: lines, arcs, dots
// Level 1: Glyphs        — letters, digits (composed from primitives)
// Level 2: Symbols       — shapes: circle, triangle, star, heart
// Level 3: Objects       — cat, house, tree (composed from shapes)
// Level 4: Compositions  — scenes with multiple objects, text labels
// Level 5: Variations    — same objects at different positions, scales, rotations
// Level 6: Abstractions  — random connected stroke patterns (novel forms)
// Level ∞: Combinations of all the above, seeded

function seededRng(seed) {
  // Simple deterministic PRNG (mulberry32)
  let s = seed | 0;
  return function () {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pick(rng, arr) { return arr[Math.floor(rng() * arr.length)]; }
function range(rng, lo, hi) { return lo + rng() * (hi - lo); }

const BRAILLE_WORDS = [
  'hello', 'world', 'fix', 'feat', 'init', 'test', 'draw', 'art',
  'cat', 'dog', 'sun', 'moon', 'star', 'tree', 'home', 'love',
  'code', 'run', 'fast', 'slow', 'big', 'tiny', 'red', 'blue',
  'open', 'push', 'pull', 'merge', 'ship', 'done', 'next', 'zero',
];

const SHAPE_NAMES = Object.keys(SHAPES);

function kernel(level, seed) {
  const rng = seededRng(seed);
  const id = `k${level}-${seed}`;

  if (level === 0) {
    // Primitive: 2–4 random lines/arcs
    const nLines = 2 + Math.floor(rng() * 3);
    const strokes = [];
    for (let i = 0; i < nLines; i++) {
      const x1 = range(rng, 0.1, 0.9);
      const y1 = range(rng, 0.15, 0.85);
      const x2 = range(rng, 0.1, 0.9);
      const y2 = range(rng, 0.15, 0.85);
      if (rng() > 0.5) {
        const mx = range(rng, 0.1, 0.9);
        const my = range(rng, 0.15, 0.85);
        strokes.push([[x1, y1], [mx, my], [x2, y2]]);
      } else {
        strokes.push([[x1, y1], [x2, y2]]);
      }
    }
    return {
      name: id, level,
      description: `Primitive: ${nLines} strokes`,
      draw: () => ({ type: 'raw', strokes }),
    };
  }

  if (level === 1) {
    // Braille word
    const word = pick(rng, BRAILLE_WORDS);
    const startX = range(rng, 0.05, 0.35);
    const startY = range(rng, 0.2, 0.5);
    const cellW = range(rng, 0.06, 0.09);
    const cellH = cellW * 2.3;  // 8-dot: 4 rows, taller cells
    return {
      name: id, level,
      description: `Braille: "${word}"`,
      draw: () => ({ type: 'raw', strokes: brailleText(word, startX, startY, cellW, cellH) }),
    };
  }

  if (level === 2) {
    // Symbol: random shape
    const shape = pick(rng, ['circle', 'triangle', 'square', 'star', 'heart']);
    const cx = range(rng, 0.25, 0.75);
    const cy = range(rng, 0.25, 0.75);
    const r = range(rng, 0.08, 0.22);
    return {
      name: id, level,
      description: `Symbol: ${shape}`,
      draw: () => ({ type: 'shapes', shapes: [{ shape, cx, cy, r }] }),
    };
  }

  if (level === 3) {
    // Object: random complex shape
    const shape = pick(rng, ['cat', 'house', 'tree']);
    const cx = range(rng, 0.3, 0.7);
    const cy = range(rng, 0.3, 0.65);
    const r = range(rng, 0.15, 0.3);
    return {
      name: id, level,
      description: `Object: ${shape}`,
      draw: () => ({ type: 'shapes', shapes: [{ shape, cx, cy, r }] }),
    };
  }

  if (level === 4) {
    // Composition: multiple objects + optional label
    const n = 2 + Math.floor(rng() * 3); // 2–4 objects
    const shapes = [];
    const usedRegions = [];
    for (let i = 0; i < n; i++) {
      let cx, cy, tries = 0;
      do {
        cx = range(rng, 0.15, 0.85);
        cy = range(rng, 0.2, 0.75);
        tries++;
      } while (tries < 10 && usedRegions.some(([ux, uy]) =>
        Math.abs(ux - cx) < 0.2 && Math.abs(uy - cy) < 0.2
      ));
      usedRegions.push([cx, cy]);
      const shape = pick(rng, SHAPE_NAMES);
      const r = range(rng, 0.06, 0.15);
      shapes.push({ shape, cx, cy, r });
    }
    const label = rng() > 0.5 ? pick(rng, BRAILLE_WORDS) : null;
    const desc = shapes.map(s => s.shape).join(' + ') + (label ? ` labeled "${label}"` : '');
    return {
      name: id, level,
      description: `Composition: ${desc}`,
      draw: () => {
        const plan = { type: 'multi', parts: [{ type: 'shapes', shapes }] };
        if (label) plan.parts.push({ type: 'raw', strokes: brailleText(label, 0.1, 0.78, 0.05, 0.11) });
        return plan;
      },
    };
  }

  if (level === 5) {
    // Variation: take a level-3 lesson and apply transforms
    const base = kernel(3, seed * 7 + 13);
    const scale = range(rng, 0.5, 1.5);
    const offsetX = range(rng, -0.15, 0.15);
    const offsetY = range(rng, -0.15, 0.15);
    const basePlan = base.draw();
    return {
      name: id, level,
      description: `Variation: ${base.description} (scale=${scale.toFixed(2)}, offset=[${offsetX.toFixed(2)},${offsetY.toFixed(2)}])`,
      draw: () => ({
        type: 'shapes',
        shapes: basePlan.shapes.map(s => ({
          ...s,
          cx: Math.max(0.15, Math.min(0.85, s.cx + offsetX)),
          cy: Math.max(0.2, Math.min(0.8, s.cy + offsetY)),
          r: Math.max(0.05, Math.min(0.3, s.r * scale)),
        })),
      }),
    };
  }

  // Level 6+: Abstractions — novel stroke patterns
  const nStrokes = 3 + Math.floor(rng() * (level - 3));
  const strokes = [];
  for (let i = 0; i < nStrokes; i++) {
    const pts = [];
    const nPts = 3 + Math.floor(rng() * 6);
    let x = range(rng, 0.1, 0.9);
    let y = range(rng, 0.15, 0.85);
    for (let p = 0; p < nPts; p++) {
      pts.push([x, y]);
      x = Math.max(0.05, Math.min(0.95, x + range(rng, -0.15, 0.15)));
      y = Math.max(0.1, Math.min(0.9, y + range(rng, -0.15, 0.15)));
    }
    strokes.push(pts);
  }
  return {
    name: id, level,
    description: `Abstract form: ${nStrokes} strokes, level ${level} complexity`,
    draw: () => ({ type: 'raw', strokes }),
  };
}

// Generate N lessons from the kernel, spreading across levels
function generateCurriculum(count, startSeed = 0) {
  const lessons = [];
  for (let i = 0; i < count; i++) {
    const seed = startSeed + i;
    // Spread across levels: mostly 1–5, occasionally 0 and 6+
    const level = seed % 13 === 0 ? 0
      : seed % 11 === 0 ? 6 + Math.floor(i / 20)
      : 1 + (seed % 5);
    lessons.push(kernel(level, seed));
  }
  return lessons;
}

module.exports = {
  FONT, SHAPES, BRAILLE, CURRICULUM,
  drawStrokes, drawText, drawShape,
  brailleText, brailleCell, dotPositions, tokenizeBraille,
  executeLesson, kernel, generateCurriculum, seededRng,
};
