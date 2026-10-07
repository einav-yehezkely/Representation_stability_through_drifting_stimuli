import { defineConfig } from "@playwright/test";

// Real-browser smoke tests. Uses the locally installed Google Chrome (no browser download).
// - production build (participant mode) served by `vite preview` on :4173
// - Vite dev server (development mode via ?dev=1) on :5173
export default defineConfig({
  testDir: "tests/e2e",
  outputDir: "test-results",
  timeout: 120000,
  workers: 1,
  reporter: [["list"]],
  use: { channel: "chrome", headless: true, acceptDownloads: true },
  webServer: [
    { command: "npx vite preview --port 4173 --strictPort", url: "http://localhost:4173/index.html", reuseExistingServer: false, timeout: 60000 },
    { command: "npx vite --port 5173 --strictPort", url: "http://localhost:5173/index.html", reuseExistingServer: false, timeout: 60000 },
  ],
});
