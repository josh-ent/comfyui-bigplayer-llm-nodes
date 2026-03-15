import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const STATE_PATH = path.join(os.tmpdir(), "bigplayer-playwright-ui-state.json");

export function readServerState() {
  if (!fs.existsSync(STATE_PATH)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
}

export function writeServerState(state) {
  fs.writeFileSync(STATE_PATH, JSON.stringify(state), "utf8");
}

export function clearServerState() {
  if (!fs.existsSync(STATE_PATH)) {
    return;
  }
  fs.unlinkSync(STATE_PATH);
}
