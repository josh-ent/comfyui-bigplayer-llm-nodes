import { defineConfig } from "@playwright/test";

const port = Number(process.env.BIGPLAYER_COMFYUI_UI_TEST_PORT ?? "18188");

export default defineConfig({
  testDir: "./tests/ui",
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  reporter: "line",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "python tests/ui/serve_comfyui.py",
    url: `http://127.0.0.1:${port}/object_info`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    gracefulShutdown: {
      signal: "SIGTERM",
      timeout: 2_000,
    },
    env: {
      BIGPLAYER_COMFYUI_UI_TEST_PORT: String(port),
    },
  },
});
