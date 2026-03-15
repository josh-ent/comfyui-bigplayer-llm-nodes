import { execFileSync } from "node:child_process";
import { clearServerState } from "./server_state.mjs";

export default async function globalTeardown() {
  try {
    execFileSync("python", ["tests/ui/cleanup_comfyui.py"], {
      stdio: "inherit",
    });
  } finally {
    clearServerState();
  }
}
