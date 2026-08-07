const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    // Use the already-running HTTP server
    baseURL: 'http://localhost:8766',
    permissions: ['camera'],
    launchOptions: {
      slowMo: process.env.SLOW ? parseInt(process.env.SLOW) : 0,
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
      ],
    },
  },
});
