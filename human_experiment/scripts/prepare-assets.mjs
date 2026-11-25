// Prepare browser-servable assets for the human experiment: one output per image set (the admin
// page switches between them; see stimulusSets in src/config/experimentConfig.js).
//
// READ-ONLY inputs (never modified), per set:
//   original: ../pca_top2_filtered_female_vgg_1.csv + ../female_faces/<filename>
//   morph:    ../pca_top2_female_generated_vgg.csv  + ../female_generated/<filename>
//   (CSV without header: filename, PC1, PC2)
//
// Outputs (all inside human_experiment/public/assets/), per set:
//   <facesDir>/<filename>     copies of every image listed in the PCA file (copies of images no
//                             longer listed are removed, so the served set matches the file)
//   <pcaJson>                 PCA coordinates in source row order
//   <manifest>                version information (hashes, counts)
//
// The two coordinates are stored as pc1/pc2 (first/second axis of the stimulus plane) whatever
// components the plane uses; `axes` names them for the researcher plot.
//
//   node scripts/prepare-assets.mjs            (all sets)
//   node scripts/prepare-assets.mjs morph      (only the named sets)

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, "..");
const repoRoot = path.resolve(projectDir, "..");
const OUT_DIR = path.join(projectDir, "public", "assets");

// Source files of each image set, relative to the repository root.
// TO REPLACE THE MORPH IMAGES: change `pcaFile` and `facesSource` of `morph` below (or put the new
// files under the same names), then `npm run prepare-assets` and `npm run build`. The number of
// images, their size and the names shown on the admin page all follow from the files.
// facesDir / pcaJson / manifest are the served names (stimulusSets in src/config/experimentConfig.js
// and STIMULUS_SET_MANIFESTS in public/admin_api.php); they do not need to change.
// The original PCA file is also used by scripts/reference/make_parity_fixtures.py.
// pca_PC3_PC4_filtered.csv (PC3/PC4) was tried for the original set on 2026-10-28 and reverted.
const SETS = {
  original: { pcaFile: "pca_top2_filtered_female_vgg_1.csv", axes: ["PC1", "PC2"], facesSource: "female_faces", facesDir: "faces", pcaJson: "pca.json", manifest: "assets-manifest.json" },
  // 2026-11-24: the morph set is the four PCA-edge women of composites/pca_edge_morph.py (one noisy
  // variant around each of the 360 morph frames + 100 noisy variants around 0 deg and 180 deg),
  // copied from composites/generated/pca_edges/experiment_set*. The previous morph set (seeds 38/46,
  // female_generated/) is kept in human_experiment/stimulus_backups/morph_38_46/; to restore it use
  //   pcaFile: "human_experiment/stimulus_backups/morph_38_46/pca_top2_female_generated_vgg.csv",
  //   facesSource: "human_experiment/stimulus_backups/morph_38_46/faces"
  morph: { pcaFile: "pca_top2_women_generated_4_corners_morph.csv", axes: ["PC1", "PC2"], facesSource: "women_generated_4_corners_morph",facesDir: "faces_morph", pcaJson: "pca_morph.json", manifest: "assets-manifest_morph.json" },
};

function sha256(bufferOrString) {
  return createHash("sha256").update(bufferOrString).digest("hex");
}

function fail(message) {
  console.error(`prepare-assets: ${message}`);
  process.exit(1);
}

/** Width and height in px of a PNG or JPEG file (read from its header), or null. */
function imageSize(file) {
  const fd = fs.openSync(file, "r");
  try {
    const head = Buffer.alloc(64 * 1024);
    const n = fs.readSync(fd, head, 0, head.length, 0);
    const b = head.subarray(0, n);
    // PNG: signature, then the IHDR chunk with width and height (big-endian).
    if (b.length >= 24 && b.readUInt32BE(0) === 0x89504e47) return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
    // JPEG: walk the segments up to a start-of-frame marker (SOF0..SOF15 except DHT/JPG/DAC).
    if (b.length >= 4 && b[0] === 0xff && b[1] === 0xd8) {
      let i = 2;
      while (i + 9 < b.length) {
        if (b[i] !== 0xff) return null;
        const marker = b[i + 1];
        if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return { width: b.readUInt16BE(i + 7), height: b.readUInt16BE(i + 5) };
        i += 2 + b.readUInt16BE(i + 2);
      }
    }
    return null;
  } finally {
    fs.closeSync(fd);
  }
}

function prepare(name, set) {
  const pcaSource = path.join(repoRoot, set.pcaFile);
  const facesSource = path.join(repoRoot, set.facesSource);
  const outFaces = path.join(OUT_DIR, set.facesDir);

  // Guard: never write outside human_experiment/.
  for (const target of [OUT_DIR, outFaces]) {
    if (!path.resolve(target).startsWith(projectDir + path.sep)) fail(`refusing to write outside human_experiment/: ${target}`);
  }
  if (!fs.existsSync(pcaSource)) fail(`[${name}] missing PCA file: ${pcaSource}`);
  if (!fs.existsSync(facesSource)) fail(`[${name}] missing faces directory: ${facesSource}`);

  const pcaBytes = fs.readFileSync(pcaSource);
  const lines = pcaBytes.toString("utf8").split(/\r?\n/).filter((line) => line.length > 0);

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
    const [file, xs, ys] = parts;
    const x = Number(xs);
    const y = Number(ys);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      problems.push(`row ${row + 1}: non-finite coordinates`);
      return;
    }
    if (seen.has(file)) {
      problems.push(`row ${row + 1}: duplicate filename ${file}`);
      return;
    }
    seen.add(file);
    imageIds.push(file);
    pc1.push(x);
    pc2.push(y);
  });

  if (problems.length) fail(`[${name}] invalid PCA file:\n  ${problems.slice(0, 20).join("\n  ")}`);

  const missing = imageIds.filter((file) => !fs.existsSync(path.join(facesSource, file)));
  if (missing.length) {
    fail(`[${name}] ${missing.length} images listed in the PCA file are missing from ${set.facesSource}/:\n  ${missing.slice(0, 20).join("\n  ")}`);
  }

  // All images of a set must have the same size (the trial screen has a fixed frame).
  const sizes = new Map(); // "WxH" -> example file
  for (const file of imageIds) {
    const size = imageSize(path.join(facesSource, file));
    if (!size) fail(`[${name}] ${file}: not a readable PNG or JPEG image`);
    const key = `${size.width}x${size.height}`;
    if (!sizes.has(key)) sizes.set(key, file);
  }
  if (sizes.size !== 1) {
    fail(`[${name}] the images do not all have the same size: ${[...sizes].map(([k, f]) => `${k} (e.g. ${f})`).join(", ")}`);
  }
  const [imageWidth, imageHeight] = [...sizes.keys()][0].split("x").map(Number);

  fs.mkdirSync(outFaces, { recursive: true });

  let copied = 0;
  let skipped = 0;
  const imageListHash = createHash("sha256");
  for (const file of imageIds) {
    const src = path.join(facesSource, file);
    const dst = path.join(outFaces, file);
    const srcSize = fs.statSync(src).size;
    if (fs.existsSync(dst) && fs.statSync(dst).size === srcSize) {
      skipped += 1;
    } else {
      fs.copyFileSync(src, dst);
      copied += 1;
    }
    imageListHash.update(`${file}:${srcSize}\n`);
  }

  // Remove copies of images that are no longer in the PCA file (only inside this set's folder).
  const listed = new Set(imageIds);
  let removed = 0;
  for (const file of fs.readdirSync(outFaces)) {
    if (!listed.has(file)) {
      fs.unlinkSync(path.join(outFaces, file));
      removed += 1;
    }
  }

  const pcaSha256 = sha256(pcaBytes);
  const manifest = {
    generatedAt: new Date().toISOString(),
    stimulusSet: name,
    pcaSource: set.pcaFile,
    pcaAxes: set.axes,
    pcaSha256,
    pcaRowCount: imageIds.length,
    facesSource: `${set.facesSource}/`,
    facesCount: imageIds.length,
    facesListSha256: imageListHash.digest("hex"),
    imageWidth,
    imageHeight,
  };

  const pcaJson = {
    source: manifest.pcaSource,
    sha256: pcaSha256,
    columns: ["imageId", "pc1", "pc2"],
    axes: set.axes, // components behind pc1 / pc2
    imageIds,
    pc1,
    pc2,
  };

  fs.writeFileSync(path.join(OUT_DIR, set.pcaJson), JSON.stringify(pcaJson));
  fs.writeFileSync(path.join(OUT_DIR, set.manifest), JSON.stringify(manifest, null, 2) + "\n");

  console.log(`[${name}] PCA rows: ${imageIds.length} (sha256 ${pcaSha256}), images ${imageWidth}x${imageHeight} px`);
  console.log(`[${name}] Faces copied: ${copied}, already present: ${skipped}, removed (no longer listed): ${removed}`);
  console.log(`[${name}] Wrote ${path.relative(projectDir, OUT_DIR)}/${set.pcaJson} and ${set.manifest}, faces in ${set.facesDir}/`);
}

const requested = process.argv.slice(2);
for (const name of requested) if (!(name in SETS)) fail(`unknown image set "${name}" (known: ${Object.keys(SETS).join(", ")})`);
for (const name of requested.length ? requested : Object.keys(SETS)) prepare(name, SETS[name]);
