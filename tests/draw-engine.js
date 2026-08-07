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
    [[cx - r * 0.08, cy + r * 0.3], [cx - r * 0.08, cy + r], [cx + r * 0.08, cy + r], [cx + r * 0.08, cy + r * 0.3]],
    // canopy layers
    ...SHAPES.triangle(cx, cy - r * 0.1, r * 0.5),
    ...SHAPES.triangle(cx, cy + r * 0.1, r * 0.6),
    ...SHAPES.triangle(cx, cy + r * 0.3, r * 0.7),
  ],
};

// ─── Drawing Curriculum ───
// Each lesson generates training data for the AI
const CURRICULUM = [
  {
    name: 'alphabet',
    description: 'Write the full alphabet',
    draw: () => ({ type: 'text', text: 'abcdefghijklm', row: 0 }),
  },
  {
    name: 'alphabet-2',
    description: 'Write the rest of the alphabet',
    draw: () => ({ type: 'text', text: 'nopqrstuvwxyz', row: 0 }),
  },
  {
    name: 'commit-feat',
    description: 'Handwrite a commit message',
    draw: () => ({ type: 'text', text: 'feat: init' }),
  },
  {
    name: 'commit-fix',
    description: 'Handwrite a bug fix commit',
    draw: () => ({ type: 'text', text: 'fix: null ref' }),
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
async function drawStrokes(page, canvasBox, strokes, speed = 3) {
  const penLift = process.env.HEADED ? 30 : 0;  // ms between strokes when visible
  for (const stroke of strokes) {
    if (stroke.length < 2) continue;
    const toX = (nx) => canvasBox.x + nx * canvasBox.width;
    const toY = (ny) => canvasBox.y + ny * canvasBox.height;

    await page.mouse.move(toX(stroke[0][0]), toY(stroke[0][1]));
    await page.mouse.down();
    for (let pi = 1; pi < stroke.length; pi++) {
      const prev = stroke[pi - 1], pt = stroke[pi];
      for (let s = 1; s <= speed; s++) {
        const t = s / speed;
        await page.mouse.move(
          toX(prev[0] + (pt[0] - prev[0]) * t),
          toY(prev[1] + (pt[1] - prev[1]) * t),
        );
      }
    }
    await page.mouse.up();
    if (penLift) await new Promise(r => setTimeout(r, penLift));
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

const WORDS = [
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
    // Glyph: random word
    const word = pick(rng, WORDS);
    const startX = range(rng, 0.05, 0.4);
    const startY = range(rng, 0.25, 0.6);
    return {
      name: id, level,
      description: `Glyph: write "${word}"`,
      draw: () => ({ type: 'text', text: word, startX, startY }),
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
    const label = rng() > 0.5 ? pick(rng, WORDS) : null;
    const desc = shapes.map(s => s.shape).join(' + ') + (label ? ` labeled "${label}"` : '');
    return {
      name: id, level,
      description: `Composition: ${desc}`,
      draw: () => {
        const plan = { type: 'multi', parts: [{ type: 'shapes', shapes }] };
        if (label) plan.parts.push({ type: 'text', text: label, startX: 0.1, startY: 0.88 });
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
  FONT, SHAPES, CURRICULUM,
  drawStrokes, drawText, drawShape, executeLesson,
  kernel, generateCurriculum, seededRng,
};
