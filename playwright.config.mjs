import { defineConfig } from "@playwright/test";
import net from "node:net";
import { readServerState, writeServerState } from "./tests/ui/server_state.mjs";

async function findFreePort() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : null;
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve(port);
      });
    });
  });
}

async function resolvePort() {
  if (process.env.BIGPLAYER_COMFYUI_UI_TEST_PORT) {
    return Number(process.env.BIGPLAYER_COMFYUI_UI_TEST_PORT);
  }

  const existingState = readServerState();
  if (existingState?.port) {
    return Number(existingState.port);
  }

  const port = await findFreePort();
  writeServerState({ port });
  return port;
}

const port = await resolvePort();

export default defineConfig({
  testDir: "./tests/ui",
  globalTeardown: "./tests/ui/global_teardown.mjs",
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
    reuseExistingServer: false,
    timeout: 180_000,
    gracefulShutdown: {
      signal: "SIGTERM",
      timeout: 10_000,
    },
    env: {
      BIGPLAYER_COMFYUI_UI_TEST_PORT: String(port),
    },
  },
});
