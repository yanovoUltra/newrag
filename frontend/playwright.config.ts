import { defineConfig, devices } from '@playwright/test'

const devPort = process.env.PLAYWRIGHT_PORT ?? '5173'
const baseURL = `http://localhost:${devPort}`

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 2 : 0,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: `npm run dev -- --port ${devPort}`,
    url: baseURL,
    reuseExistingServer: true,
    timeout: 60_000,
  },
})
