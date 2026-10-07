// Verify that no pre-existing repository file outside human_experiment/ changed
// and that every new file lives under human_experiment/ (PRD §2, acceptance check 1).
//
//   node scripts/verify-repo-unchanged.mjs --write-baseline   (run once, before work)
//   node scripts/verify-repo-unchanged.mjs                    (compare)
//
// Checks:
//   1. No tracked file differs from the baseline commit (git diff against it is empty),
//      except tracked files that the baseline itself already showed as modified.
//   2. Every untracked file is under human_experiment/.
//   3. Ignored entries outside human_experiment/ (the root .gitignore hides *.csv, *.png, ...)
//      are exactly those present at baseline, with unchanged size and mtime.
//   4. The PCA source file has the same SHA-256 as at baseline.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(here, "..");
const repoRoot = path.resolve(projectDir, "..");
const BASELINE = path.join(here, "repo-baseline.json");
const PROJECT_PREFIX = "human_experiment/";

function git(args) {
  return execFileSync("git", args, { cwd: repoRoot, encoding: "utf8", maxBuffer: 1 << 28 });
}

function porcelain(extra) {
  return git(["status", "--porcelain=v1", ...extra])
    .split("\n")
    .filter(Boolean)
    .map((line) => ({ code: line.slice(0, 2), file: line.slice(3).replace(/^"|"$/g, "") }));
}

function ignoredOutsideProject() {
  return porcelain(["--ignored"])
    .filter((entry) => entry.code === "!!" && !entry.file.startsWith(PROJECT_PREFIX))
    .map((entry) => entry.file)
    .sort();
}

function statSignature(rel) {
  const abs = path.join(repoRoot, rel);
  const stat = fs.statSync(abs);
  return stat.isDirectory() ? { dir: true, mtimeMs: stat.mtimeMs } : { size: stat.size, mtimeMs: stat.mtimeMs };
}

function sha256File(rel) {
  return createHash("sha256").update(fs.readFileSync(path.join(repoRoot, rel))).digest("hex");
}

function snapshot() {
  const ignored = ignoredOutsideProject();
  return {
    head: git(["rev-parse", "HEAD"]).trim(),
    modifiedTracked: porcelain([]).filter((e) => e.code !== "??").map((e) => e.file).sort(),
    ignored: Object.fromEntries(ignored.map((rel) => [rel, statSignature(rel)])),
    pcaSha256: sha256File("pca_top2_filtered_female_vgg_1.csv"),
  };
}

if (process.argv.includes("--write-baseline")) {
  const snap = { capturedAt: new Date().toISOString(), ...snapshot() };
  fs.writeFileSync(BASELINE, JSON.stringify(snap, null, 2) + "\n");
  console.log(`Baseline written: HEAD ${snap.head}, ${Object.keys(snap.ignored).length} ignored entries outside human_experiment/.`);
  process.exit(0);
}

if (!fs.existsSync(BASELINE)) {
  console.error("No baseline found. Run with --write-baseline before starting work.");
  process.exit(2);
}

const baseline = JSON.parse(fs.readFileSync(BASELINE, "utf8"));
const errors = [];

// 1. Tracked files: compare working tree with the baseline commit.
// Files the user explicitly asked to change are listed in baseline.userAuthorizedChanges
// (path -> reason) and reported, not treated as violations.
const authorized = baseline.userAuthorizedChanges ?? {};
const changedSinceBaseline = git(["diff", "--name-only", baseline.head])
  .split("\n")
  .filter(Boolean)
  .filter((file) => !baseline.modifiedTracked.includes(file));
for (const file of changedSinceBaseline) {
  if (file in authorized) console.log(`authorized change: ${file} (${authorized[file]})`);
  else errors.push(`tracked file changed: ${file}`);
}

// 2. Untracked files must be inside human_experiment/.
for (const entry of porcelain(["-uall"])) {
  if (entry.code === "??" && !entry.file.startsWith(PROJECT_PREFIX)) {
    errors.push(`new file outside human_experiment/: ${entry.file}`);
  }
}

// 3. Ignored entries outside human_experiment/.
const nowIgnored = ignoredOutsideProject();
for (const rel of nowIgnored) {
  if (!(rel in baseline.ignored)) {
    errors.push(`new ignored entry outside human_experiment/: ${rel}`);
    continue;
  }
  const before = baseline.ignored[rel];
  const after = statSignature(rel);
  if (!before.dir && (before.size !== after.size || before.mtimeMs !== after.mtimeMs)) {
    errors.push(`ignored file changed: ${rel}`);
  }
}
for (const rel of Object.keys(baseline.ignored)) {
  if (!nowIgnored.includes(rel)) errors.push(`ignored entry removed: ${rel}`);
}

// 4. PCA source content.
if (sha256File("pca_top2_filtered_female_vgg_1.csv") !== baseline.pcaSha256) {
  errors.push("pca_top2_filtered_female_vgg_1.csv content changed");
}

const head = git(["rev-parse", "HEAD"]).trim();
console.log(`Baseline HEAD: ${baseline.head}`);
console.log(`Current  HEAD: ${head}${head === baseline.head ? "" : " (new commits since baseline; compared against the baseline commit)"}`);
if (errors.length) {
  console.error(`FAILED (${errors.length}):\n  ${errors.join("\n  ")}`);
  process.exit(1);
}
console.log("OK: no pre-existing file outside human_experiment/ changed; all new files are under human_experiment/.");
