const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');
const { CURRICULUM, executeLesson, generateCurriculum, kernel } = require('./draw-engine');

const RECORDINGS_DIR = path.resolve(__dirname, '..', 'recordings');

async function waitFor(fn, timeout = 10000, interval = 200) {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    if (fn()) return true;
    await new Promise(r => setTimeout(r, interval));
  }
  return fn();
}

// ─── Pick which lessons to run ───
//
//   CURRICULUM=commit-feat          Single named lesson (default)
//   CURRICULUM=all                  All 9 hand-crafted lessons
//   CURRICULUM=shapes               Lessons matching prefix
//   CURRICULUM=kernel:20            20 kernel-generated lessons (seed 0)
//   CURRICULUM=kernel:50:1000       50 lessons starting at seed 1000
//   CURRICULUM=level:3:42           Single kernel lesson at level 3, seed 42
//
const FILTER = process.env.CURRICULUM || 'commit-feat';

let lessons;
if (FILTER === 'all') {
  lessons = CURRICULUM;
} else if (FILTER.startsWith('kernel:')) {
  const parts = FILTER.split(':');
  const count = parseInt(parts[1]) || 10;
  const seed = parseInt(parts[2]) || 0;
  lessons = generateCurriculum(count, seed);
} else if (FILTER.startsWith('level:')) {
  const parts = FILTER.split(':');
  const level = parseInt(parts[1]) || 1;
  const seed = parseInt(parts[2]) || 0;
  lessons = [kernel(level, seed)];
} else {
  lessons = CURRICULUM.filter(l => l.name === FILTER || l.name.startsWith(FILTER));
}

test.describe('AI Drawing Curriculum', () => {

  for (const lesson of lessons) {
    test(`lesson: ${lesson.name} — ${lesson.description}`, async ({ browser }) => {
      const t0 = Date.now();
      const elapsed = () => `${((Date.now() - t0) / 1000).toFixed(1)}s`;

      // Clean recordings before each lesson
      if (fs.existsSync(RECORDINGS_DIR)) {
        for (const f of fs.readdirSync(RECORDINGS_DIR)) {
          if (f.endsWith('.webm')) fs.unlinkSync(path.join(RECORDINGS_DIR, f));
        }
      }

      // --- Viewer (records the drawing) ---
      const ctx = await browser.newContext({ permissions: ['camera'] });
      const viewerPage = await ctx.newPage();
      const consoleLogs = [];
      viewerPage.on('console', (msg) => consoleLogs.push(msg.text()));
      await viewerPage.goto('/viewer');

      // Accept GDPR consent
      await viewerPage.locator('#consentAccept').click();
      await expect(viewerPage.locator('#dot')).toHaveClass(/ok/, { timeout: 3000 });

      // --- Drawer (does the drawing) ---
      const drawerPage = await ctx.newPage();
      await drawerPage.goto('/');
      await expect(drawerPage.locator('#dot')).toHaveClass(/ok/, { timeout: 3000 });
      console.log(`[${elapsed()}] Connected — starting lesson: ${lesson.name}`);

      // --- Execute the drawing lesson ---
      const canvasBox = await drawerPage.locator('#c').boundingBox();
      const drawStart = Date.now();
      await executeLesson(drawerPage, canvasBox, lesson);
      const drawMs = Date.now() - drawStart;
      console.log(`[${elapsed()}] Drew "${lesson.name}" in ${drawMs}ms`);

      // --- Verify recording lifecycle ---
      const slowScale = process.env.SLOW ? 5 : 1;
      expect(await waitFor(() =>
        consoleLogs.some(l => l.includes('Recording started')),
        10000 * slowScale
      )).toBe(true);
      console.log(`[${elapsed()}] Recording started`);

      expect(await waitFor(() =>
        consoleLogs.some(l => l.includes('Recording stopped')),
        8000 * slowScale, 100
      )).toBe(true);
      console.log(`[${elapsed()}] Recording stopped (5s idle)`);

      expect(await waitFor(() =>
        consoleLogs.some(l => l.includes('Recording saved to server')),
        3000 * slowScale, 100
      )).toBe(true);

      // --- Verify file saved ---
      expect(await waitFor(() =>
        fs.existsSync(RECORDINGS_DIR) &&
        fs.readdirSync(RECORDINGS_DIR).some(f => f.endsWith('.webm')),
        2000, 100
      )).toBe(true);

      const recordings = fs.readdirSync(RECORDINGS_DIR).filter(f => f.endsWith('.webm'));
      const stats = fs.statSync(path.join(RECORDINGS_DIR, recordings[0]));
      expect(stats.size).toBeGreaterThan(0);

      const totalMs = Date.now() - t0;
      console.log(`[${elapsed()}] ✓ ${lesson.name}: saved ${recordings[0]} (${(stats.size / 1024).toFixed(1)} KB) — total ${(totalMs/1000).toFixed(1)}s`);
      if (!process.env.SLOW && !lesson.name.startsWith('math-')) expect(totalMs).toBeLessThan(20000);

      await drawerPage.close();
      await ctx.close();
    });
  }
});


// ─── GDPR Consent E2E Tests ───
test.describe('GDPR Consent', () => {

  test('consent modal blocks UI until accepted', async ({ browser }) => {
    const ctx = await browser.newContext({ permissions: ['camera'] });
    const page = await ctx.newPage();
    await page.goto('/viewer');

    // Consent overlay should be visible
    await expect(page.locator('#consentOverlay')).toBeVisible();
    // Canvas should be hidden
    await expect(page.locator('#c')).toBeHidden();
    // Camera PiP should be hidden
    await expect(page.locator('#camWrap')).toBeHidden();

    // Accept consent
    await page.locator('#consentAccept').click();

    // Overlay hidden, canvas visible
    await expect(page.locator('#consentOverlay')).toBeHidden();
    await expect(page.locator('#c')).toBeVisible();

    await ctx.close();
  });

  test('declining consent blocks the session', async ({ browser }) => {
    const ctx = await browser.newContext({ permissions: ['camera'] });
    const page = await ctx.newPage();
    const consoleLogs = [];
    page.on('console', (msg) => consoleLogs.push(msg.text()));
    await page.goto('/viewer');

    await expect(page.locator('#consentOverlay')).toBeVisible();
    await page.locator('#consentDecline').click();

    // Page should show declined message
    await expect(page.locator('body')).toContainText('Consent declined');
    // Canvas and camera should not exist
    await expect(page.locator('#c')).toHaveCount(0);

    expect(consoleLogs.some(l => l.includes('GDPR consent declined'))).toBe(true);

    await ctx.close();
  });

  test('consent is logged to backend', async ({ browser }) => {
    const ctx = await browser.newContext({ permissions: ['camera'] });
    const page = await ctx.newPage();
    await page.goto('/viewer');
    await page.locator('#consentAccept').click();
    await expect(page.locator('#consentOverlay')).toBeHidden();

    // Check consent log via API
    const resp = await page.evaluate(() =>
      fetch('/api/consent').then(r => r.json())
    );
    expect(resp.records.length).toBeGreaterThan(0);
    const last = resp.records[resp.records.length - 1];
    expect(last.action).toBe('granted');
    expect(last.gdpr_article).toBe('6(1)(a)');
    expect(last.scope).toContain('camera');
    expect(last.scope).toContain('screen_recording');

    await ctx.close();
  });

  test('studio consent flow works', async ({ browser }) => {
    const ctx = await browser.newContext({ permissions: ['camera'] });
    const page = await ctx.newPage();
    await page.goto('/studio');

    // Studio hidden, consent visible
    await expect(page.locator('#consentOverlay')).toBeVisible();
    await expect(page.locator('#studio')).toBeHidden();

    await page.locator('#consentAccept').click();

    await expect(page.locator('#consentOverlay')).toBeHidden();
    await expect(page.locator('#studio')).toBeVisible();

    // Verify consent logged
    const resp = await page.evaluate(() =>
      fetch('/api/consent').then(r => r.json())
    );
    const studioConsent = resp.records.find(r => r.scope && r.scope.includes('labels'));
    expect(studioConsent).toBeTruthy();
    expect(studioConsent.action).toBe('granted');

    await ctx.close();
  });
});
