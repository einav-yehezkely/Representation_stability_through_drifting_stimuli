// Acceptance check 13 (build part): in the participant build, development controls are absent.
//   1. config.development.enabled is false (so ?dev=1 is ignored by production builds).
//   2. dist/index.html does not load or preload the development chunks.
//   3. The development chunks are reachable only through dynamic import (never a static import
//      from the entry chunk).
//   4. The review page is not linked from the participant page.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { experimentConfig } from "../src/config/experimentConfig.js";

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(projectDir, "dist");
const errors = [];

if (experimentConfig.development.enabled !== false) errors.push("experimentConfig.development.enabled must be false for participants");
if (!fs.existsSync(path.join(dist, "index.html"))) {
  console.error("dist/index.html not found. Run `npm run build` first.");
  process.exit(2);
}

const html = fs.readFileSync(path.join(dist, "index.html"), "utf8");
if (/devPanel|devSetup/.test(html)) errors.push("dist/index.html references development chunks");
if (/review\.html/.test(html)) errors.push("dist/index.html links the review page");

const entry = html.match(/<script type="module"[^>]*src="\.\/(app\/index-[^"]+\.js)"/)?.[1];
if (!entry) errors.push("could not find the entry chunk in dist/index.html");
else {
  const code = fs.readFileSync(path.join(dist, entry), "utf8");
  const staticImports = [...code.matchAll(/^import[^;]*?from\s*"([^"]+)"/gm)].map((m) => m[1]);
  if (staticImports.some((s) => /devPanel|devSetup/.test(s))) errors.push("entry chunk statically imports development code");
  if (!/import\(\s*"\.\/devSetup-/.test(code) && !/devSetup-[\w-]+\.js/.test(code)) {
    errors.push("development setup is expected to be a lazily loaded chunk");
  }
}

if (errors.length) {
  console.error(`FAILED:\n  ${errors.join("\n  ")}`);
  process.exit(1);
}
console.log("OK: participant build has development mode disabled; development code is only lazily loaded and never linked.");
