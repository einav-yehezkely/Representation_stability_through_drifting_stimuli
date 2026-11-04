// Loads the real PCA data prepared by `npm run prepare-assets` (public/assets/pca.json).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildPca } from "../../src/scientific/pcaLoader.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, "..", "..");

export function loadRealPca() {
  const pcaPath = path.join(projectDir, "public", "assets", "pca.json");
  if (!fs.existsSync(pcaPath)) {
    throw new Error("public/assets/pca.json is missing. Run `npm run prepare-assets` first.");
  }
  const manifest = JSON.parse(fs.readFileSync(path.join(projectDir, "public", "assets", "assets-manifest.json"), "utf8"));
  return buildPca(JSON.parse(fs.readFileSync(pcaPath, "utf8")), manifest);
}

export function loadParityFixture() {
  return JSON.parse(fs.readFileSync(path.join(projectDir, "tests", "fixtures", "parity", "parity.json"), "utf8"));
}
