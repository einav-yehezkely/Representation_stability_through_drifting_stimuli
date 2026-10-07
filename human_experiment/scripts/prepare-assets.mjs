// Prepare browser-servable assets for the human experiment.
//
// READ-ONLY inputs (never modified):
//   ../pca_top2_filtered_female_vgg_1.csv   (no header: filename, PC1, PC2)
//   ../female_faces/<filename>
//
// Outputs (all inside human_experiment/public/assets/):
//   faces/<filename>          copies of every image listed in the PCA file
//   pca.json                  PCA coordinates in source row order
//   assets-manifest.json      version information (hashes, counts)

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, "..");
const repoRoot = path.resolve(projectDir, "..");

const PCA_SOURCE = path.join(repoRoot, "pca_top2_filtered_female_vgg_1.csv");
const FACES_SOURCE = path.join(repoRoot, "female_faces");
const OUT_DIR = path.join(projectDir, "public", "assets");
const OUT_FACES = path.join(OUT_DIR, "faces");

function sha256(bufferOrString) {
  return createHash("sha256").update(bufferOrString).digest("hex");
}

function fail(message) {
  console.error(`prepare-assets: ${message}`);
  process.exit(1);
}

// Guard: never write outside human_experiment/.
for (const target of [OUT_DIR, OUT_FACES]) {
  if (!path.resolve(target).startsWith(projectDir + path.sep)) {
    fail(`refusing to write outside human_experiment/: ${target}`);
  }
}

if (!fs.existsSync(PCA_SOURCE)) fail(`missing PCA file: ${PCA_SOURCE}`);
if (!fs.existsSync(FACES_SOURCE)) fail(`missing faces directory: ${FACES_SOURCE}`);

const pcaBytes = fs.readFileSync(PCA_SOURCE);
const pcaText = pcaBytes.toString("utf8");
const lines = pcaText.split(/\r?\n/).filter((line) => line.length > 0);

const imageIds = [];
const pc1 = [];
const pc2 = [];
const seen = new Set();
const problems = [];

lines.forEach((line, row) => {
  const parts = line.split(",");
  if (parts.length !== 3) {
    problems.push(`row ${row + 1}: expected 3 columns, found ${parts.length}`);
    return;
  }
  // Same normalization as the source loaders: the filename column as-is
  // (load_top2_filtered uses df.iloc[:, 0].values).
  const [name, xs, ys] = parts;
  const x = Number(xs);
  const y = Number(ys);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    problems.push(`row ${row + 1}: non-finite coordinates`);
    return;
  }
  if (seen.has(name)) {
    problems.push(`row ${row + 1}: duplicate filename ${name}`);
    return;
  }
  seen.add(name);
  imageIds.push(name);
  pc1.push(x);
  pc2.push(y);
});

if (problems.length) fail(`invalid PCA file:\n  ${problems.slice(0, 20).join("\n  ")}`);

const missing = imageIds.filter((name) => !fs.existsSync(path.join(FACES_SOURCE, name)));
if (missing.length) {
  fail(`${missing.length} images listed in the PCA file are missing from female_faces/:\n  ${missing.slice(0, 20).join("\n  ")}`);
}

fs.mkdirSync(OUT_FACES, { recursive: true });

let copied = 0;
let skipped = 0;
const imageListHash = createHash("sha256");
for (const name of imageIds) {
  const src = path.join(FACES_SOURCE, name);
  const dst = path.join(OUT_FACES, name);
  const srcSize = fs.statSync(src).size;
  if (fs.existsSync(dst) && fs.statSync(dst).size === srcSize) {
    skipped += 1;
  } else {
    fs.copyFileSync(src, dst);
    copied += 1;
  }
  imageListHash.update(`${name}:${srcSize}\n`);
}

const pcaSha256 = sha256(pcaBytes);
const manifest = {
  generatedAt: new Date().toISOString(),
  pcaSource: "pca_top2_filtered_female_vgg_1.csv",
  pcaSha256,
  pcaRowCount: imageIds.length,
  facesSource: "female_faces/",
  facesCount: imageIds.length,
  facesListSha256: imageListHash.digest("hex"),
};

const pcaJson = {
  source: manifest.pcaSource,
  sha256: pcaSha256,
  columns: ["imageId", "pc1", "pc2"],
  imageIds,
  pc1,
  pc2,
};

fs.writeFileSync(path.join(OUT_DIR, "pca.json"), JSON.stringify(pcaJson));
fs.writeFileSync(path.join(OUT_DIR, "assets-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

console.log(`PCA rows: ${imageIds.length} (sha256 ${pcaSha256})`);
console.log(`Faces copied: ${copied}, already present: ${skipped}`);
console.log(`Wrote ${path.relative(projectDir, OUT_DIR)}/pca.json and assets-manifest.json`);
