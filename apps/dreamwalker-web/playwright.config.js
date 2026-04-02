import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  timeout: 60_000,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    headless: true
  },
  webServer: [
    {
      command: 'npm run relay',
      url: 'http://127.0.0.1:8787/health',
      reuseExistingServer: true,
      timeout: 120_000
    },
    {
      command: 'npm run robotics:bridge',
      url: 'http://127.0.0.1:8790/health',
      reuseExistingServer: true,
      timeout: 120_000
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 4173',
      port: 4173,
      reuseExistingServer: true,
      timeout: 120_000
    }
  ]
});
