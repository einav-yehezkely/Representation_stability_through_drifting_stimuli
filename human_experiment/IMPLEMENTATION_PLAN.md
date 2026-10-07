# Detailed Build Plan — Human Experiment

This document is the work plan for building the website as specified in [HUMAN_EXPERIMENT_PRD.md](HUMAN_EXPERIMENT_PRD.md). The plan adds no requirements. Every scientific parameter in it is derived from the inspected source code (Section 1). Wherever the source contradicts itself or is ambiguous, the point appears in Section 2 as an open question rather than as a decision.

---

## 0. Repository protection rules (PRD §2) — apply to every stage

1. **Before starting work**: save a baseline snapshot outside the repository (in the scratchpad directory):
   - `git status --porcelain=v1 -uall`
   - the full `git diff` (currently `human_experiment/HUMAN_EXPERIMENT_PRD.md` is already marked M and the PDF is untracked; both must stay as they are)
   - `git ls-files -s` (hash of every tracked file) and the SHA-256 of `pca_top2_filtered_female_vgg_1.csv`
2. Every new file is created **only** under `human_experiment/`. Do not touch the root `.gitignore`, `.vscode/`, `.venv/`, any existing file, or `HUMAN_EXPERIMENT_PRD.md` itself.
3. `npm install`, `npm run build` and all scripts run with `cwd = human_experiment/`. Their output (`node_modules/`, `dist/`, `public/assets/`) stays inside that directory. A dedicated `human_experiment/.gitignore` is created.
4. `female_faces/` and `pca_top2_filtered_female_vgg_1.csv` are **read-only** inputs to the preparation script. Copies and derived files are written only to `human_experiment/public/assets/`.
5. Source Python code needed for parity checks is **copied** to `human_experiment/scripts/reference/`, and only the copy is used.
6. **At the end of the work**: rerun the baseline commands and compare. The only permitted changes are new files under `human_experiment/`. The result is documented in the README (acceptance check 1).

---

## 1. Source inspection findings (PRD §3) — the scientific basis

| Topic | Source | Finding |
| --- | --- | --- |
| Images | `female_faces/` | File names such as `000004.jpg`, JPEG, 178×218. All 11,817 images listed in the PCA file exist (85.7 MB in total). |
| PCA data | `pca_top2_filtered_female_vgg_1.csv` | No header; columns `filename, PC1(x), PC2(y)`. 11,817 rows, no duplicate names. Loaded by `load_top2_filtered()` ([classify_rotation_resnet50.py:236](../network_classification/classify_rotation_resnet50.py#L236)). Produced by [PCA_2_and_rest.py](../extract_embeddings/PCA_2_and_rest.py) (InceptionResNetV1, the 10% with the smallest residuals) and filtered by [filter_pca.py](../network_classification/tools/filter_pca.py). |
| Angle convention | [classify_rotation_resnet50.py:52-56](../network_classification/classify_rotation_resnet50.py#L52-L56) | `angle_deg = degrees(atan2(y, x)) % 360`, origin at (0,0). This convention **is supported by the source**, so it may be used. |
| Initial center A | `create_base_and_opposite_points(angle)` ([:255](../network_classification/classify_rotation_resnet50.py#L255)) | `target_radius = 0.45`. An **actual data point** is chosen that minimizes `angle_error + 100·|r − 0.45|`. For 0° this is `183785.jpg` = (0.44447, −0.00393), angle 359.49°, radius 0.4445. |
| Center B | Same function | `opposite_point = -base_point` (exact reflection). |
| Rotation | `rotate_vector` ([:300](../network_classification/classify_rotation_resnet50.py#L300)) | **Counterclockwise** rotation matrix (positive = CCW). The update is **incremental**: after each iteration `base_point = rotate_vector(base_point, ROTATION_DEGS)` ([:1783-1784](../network_classification/classify_rotation_resnet50.py#L1783-L1784)), after selection and training, not before them. |
| Cluster construction | `collect_nearest_images` ([:321](../network_classification/classify_rotation_resnet50.py#L321)) | The `k` nearest points by 2D Euclidean distance to the current center, sorted by distance (`argpartition` followed by `argsort`). The cluster is recomputed every iteration. There is **no** exclusion of previously shown images across iterations. |
| Cluster size | `NUM_OF_IMAGES_PER_CLUSTER` ([:1467](../network_classification/classify_rotation_resnet50.py#L1467)) | 64 in the code; 65 set in commit 350e7ce; 100 in the seminar paper (§3.4) → resolved: k=1 (D1). |
| Stimulus selection | `__main__` loop | The network receives **all** 2k images in each iteration, with equal weight. The source has no "select one face" step. |
| Initial network training | [generate_rotation_sequence.py:363-381](../network_classification/tools/generate_rotation_sequence.py#L363-L381) and `resnet50_embeddings_training.py` (`split_data`) | k=1000 around the **exact** point (0.45·cosθ, 0.45·sinθ) rather than a data point, using a different PCA file (`..._20-30percent.csv`) → not used; training uses nearest-unused (D4). |
| Trajectory sequence | `generate_rotation_sequence(... num_steps=360, rotation_range=360, used_indices=set())` ([:1491](../network_classification/classify_rotation_resnet50.py#L1491)) | 360 unique images, the nearest to the rotated base point at each degree. Used for evaluation and plots. |
| Plots | `create_prediction_scatter` ([:807](../network_classification/classify_rotation_resnet50.py#L807)), `plot_clusters_with_given_indices` ([generate_rotation_sequence.py:205](../network_classification/tools/generate_rotation_sequence.py#L205)) | All points in gray (s=5, alpha=0.3); A blue, B red; centers as a black × (base) and * (opposite); dashed circle of radius `max(r)·1.05`; radial lines and labels every 20°; x=0/y=0 axes; grid; `axis equal`; PC1/PC2 labels. |
| Randomness | [:25-29](../network_classification/classify_rotation_resnet50.py#L25-L29) | `SEED = 42`. In [merge_sequences.py](../network_classification/tools/merge_sequences.py) A/B are interleaved with probability 0.5 (not balanced blocks). |

---

## 2. Researcher decisions and remaining questions

### 2.1 Decisions made (binding)

| # | Topic | Decision | Implementation |
| --- | --- | --- | --- |
| D1 | Cluster size | **k = 1** | `scientific.clusterSize: 1`. Instead of the source's 64, the same function, `collect_nearest_images`, is used with k=1 |
| D2 | Selecting one face | **A cluster of size 1 → its single image** | Selection is **deterministic**: the image nearest (2D Euclidean distance) to the center of the selected group. There is no randomness in image selection; the only randomness is in the group order |
| D3 | Center B | **Always 180° from center A** | `B = −A` (A's data point reflected through the origin, as in the source's `opposite_point = -base_point`). Both parameters `initialAngleA/B` remain in the configuration (PRD §6), but `validateConfig` **rejects** any values for which `(initialAngleB − initialAngleA) mod 360 ≠ 180`. B's nominal angle comes from the configuration, and its actual position is `−A` |
| D4 | Selection during training | **On each trial, the image nearest to the group center that has not yet been selected** | A `usedTrainingIndices` set (like `used_indices` in the source's `generate_rotation_sequence`). On each training trial: nearest among images not in `usedTrainingIndices`. An image enters the set **only after a saved response** (on a load failure, the same image is retried) |
| D5 | Repeats in drift | **A drift image does not repeat within drift; training images may be used in drift** | A separate `usedDriftIndices` set that starts empty at the beginning of drift. On each drift trial: nearest among images not in `usedDriftIndices`. Training images are **not** excluded in drift |
| D6 | PCA file | **`pca_top2_filtered_female_vgg_1.csv`** | Unchanged. `female_facenet_vggface2_embeddings.csv` (512 dimensions, 118,164 images) is the embeddings file from which the PCA was derived ([PCA_2_and_rest.py](../extract_embeddings/PCA_2_and_rest.py)); it is not used by the website |

### 2.2 Consequences measured on the real data (Node simulation, defaults 0°/180°, 1° step)

- Center A = `183785.jpg` (359.49°, r=0.4445); therefore the first training trial from group A shows that image itself (distance 0). Center B is at 179.49°, and the nearest image to it is at distance 0.0017.
- **Training**: the distance to the center grows as the nearest images are used up. For A: 0.037 on the group's 20th trial and 0.060 on its 60th. For B: 0.018 and 0.032. The centers themselves stay fixed, but the shown images gradually move away from them.
- **Drift without repeats (D5)**: centers A and B travel along the same ring. So when center B reaches a region A has already passed through, the nearest images there have already been used, and the next one in line is chosen. In a simulation with a random, balanced group order (180/180), the distance between the image and the center was: median 0.006, 90th percentile 0.058, maximum 0.104 (center radius 0.445).
- Drift does not depend on the number of training trials, because training images are not excluded from it. The drift image sequence is therefore determined solely by the drift group order.
- The actual distance is saved in every record (`distanceToCenter`) for analysis.

### 2.3 Questions still open

| # | Question | Default until decided |
| --- | --- | --- |
| OQ1 | "Displayed" versus actual angle: the 0° center actually lies at 359.49° | `currentAngleA/B` = the nominal angle (`initial + (k−1)·step`, as in the PRD table), plus `centerA_pc1/pc2/actualAngle` columns |
| OQ2 | "Trajectory membership" for the visualization | Main layer: the images nearest to centers A and B at every drift position (per the effective configuration, without exclusion). An additional layer that can be enabled: the original `generate_rotation_sequence` sequence |
| OQ3 | UI parameters not specified in the PRD | Marked placeholders: `feedbackDurationMs: 1000`, `trainingBlockSize: 4`, image at its source size (178×218) scaled ×2 |

---

## 3. Technology stack

- **Vite + JavaScript ES modules (vanilla)**, no framework. The application is small, and the PRD asks for "practical, no unnecessary infrastructure".
- **Vitest + jsdom** for unit tests and integration tests of the engine and UI. **Playwright** (optional, a devDependency inside `human_experiment/`) for a smoke test in a real browser: onset timing and the download prompt.
- **Node scripts** for asset preparation. Python is needed only for the parity script, in an isolated venv at `human_experiment/.venv-parity/` (the root `.venv` does not include pandas and will not be touched).
- Two pages (Vite multi-page): `index.html` for the experiment, `review.html` for the researcher view.

---

## 4. File structure (per PRD §16)

```text
human_experiment/
├── .gitignore                     # node_modules, dist, public/assets/faces, .venv-parity
├── README.md
├── IMPLEMENTATION_PLAN.md         # this document
├── package.json / package-lock.json
├── vite.config.js                 # multi-page, outDir=dist (inside the directory)
├── index.html                     # experiment (participant/dev)
├── review.html                    # researcher view (visualization)
├── public/assets/
│   ├── faces/                     # image copies (created by the script, gitignored)
│   ├── pca.json                   # derived from the CSV: [{imageId, pc1, pc2}] + metadata
│   └── assets-manifest.json       # sha256 of the CSV, image count, hash of the image list
├── scripts/
│   ├── prepare-assets.mjs         # read-only on the source → public/assets
│   ├── verify-repo-unchanged.mjs  # comparison against the baseline
│   └── reference/
│       ├── source_copy.py         # verbatim copy of the source's scientific functions
│       └── make_parity_fixtures.py# produces tests/fixtures/parity/*.json
├── src/
│   ├── main.js                    # composition root: config → storage adapter → engine → UI
│   ├── review.js                  # composition root of the researcher view
│   ├── config/
│   │   ├── experimentConfig.js    # single entry point for all parameters
│   │   └── validateConfig.js
│   ├── content/
│   │   ├── consent.js             # text with [RESEARCHER: ...] placeholders
│   │   ├── instructions.js
│   │   └── uiText.js              # transition, feedback, errors, completion
│   ├── components/
│   │   ├── consentScreen.js
│   │   ├── instructionsScreen.js
│   │   ├── trialScreen.js
│   │   ├── transitionScreen.js
│   │   ├── completionScreen.js
│   │   ├── recoveryScreen.js
│   │   ├── errorBanner.js
│   │   └── devPanel.js            # loaded only in development mode
│   ├── experiment/
│   │   ├── experimentEngine.js    # state machine
│   │   ├── trainingPhase.js       # sliding-window criterion
│   │   ├── driftPhase.js          # exact movement order
│   │   ├── responseCoding.js
│   │   ├── seededRandomization.js
│   │   ├── stimulusPresenter.js   # load/decode/onset (DOM-aware, injected)
│   │   └── session.js             # sessionId, participantId, metadata
│   ├── scientific/                # no DOM/storage dependencies
│   │   ├── pcaLoader.js
│   │   ├── trajectory.js          # angleDeg, rotateVector, createBasePoint
│   │   ├── clusterConstruction.js # collectNearest(k)
│   │   └── stimulusSelector.js
│   ├── visualization/
│   │   ├── pcaPlot.js             # SVG renderer (parallel to the source's matplotlib plots)
│   │   └── plotExport.js          # SVG + PNG
│   └── services/dataStorage/
│       ├── storageContract.js
│       ├── index.js               # createStorage(config) → adapter
│       ├── localCsvStorage.js
│       └── csvSerializer.js
└── tests/
    ├── unit/ …                    # per module
    ├── integration/ …             # engine + fake presenter + memory storage
    ├── fixtures/parity/*.json
    └── e2e/ (Playwright, optional)
```

---

## 5. Build stages

Each stage ends with passing tests before moving to the next one.

### Stage 0 — Baseline and setup
1. Save the baseline per Section 0.1.
2. Manual `npm create vite` (writing `package.json` only, without a scaffold that writes to the root); install `vite`, `vitest`, `jsdom` and (`@playwright/test`).
3. A local `.gitignore`; `vite.config.js` with `root: human_experiment`, `build.outDir: 'dist'`, and `input: {index, review}`.
4. npm scripts: `prepare-assets`, `dev`, `build`, `preview`, `test`, `test:e2e`, `parity:fixtures`, `verify:repo`.

### Stage 1 — Asset preparation (`scripts/prepare-assets.mjs`)
1. Reads `../pca_top2_filtered_female_vgg_1.csv` (no header) and keeps the original number strings. Conversion uses `Number()` (float64, like numpy; a possible one-ulp difference from the pandas parser is checked by the parity test).
2. Validates: 3 columns, no duplicate names, finite values, and a file for every name in `../female_faces/`. If anything is missing, the script fails with the list of missing files rather than skipping them silently.
3. Copies the 11,817 images to `public/assets/faces/`. The entire pool is copied, not just the clusters, because `degreesPerTrial` and k are configurable.
4. Writes `pca.json` and `assets-manifest.json` (CSV sha256, row count, hash of the image list, creation date). These version strings go into the session metadata.

### Stage 2 — Central configuration (`src/config/experimentConfig.js`)
```js
export const experimentConfig = {
  experimentVersion: "1.0.0",
  schemaVersion: "1",
  training: { initialAngleA: 0, initialAngleB: 180, minTrials: 20, accuracyWindow: 20, accuracyThreshold: 0.80 },
  drift:    { numberOfTrials: 360, degreesPerTrial: 1 },
  assets:   { facesDirectory: "assets/faces/", pcaDataPath: "assets/pca.json", manifestPath: "assets/assets-manifest.json" },
  scientific: {
    // derived from network_classification/classify_rotation_resnet50.py
    origin: [0, 0],                    // atan2 around origin (source line 52-56)
    angleConvention: "atan2(PC2,PC1) mod 360, degrees",
    rotationDirection: "counterclockwise",   // rotate_vector, source line 300
    targetRadius: 0.45,                // create_base_and_opposite_points, line 276
    radiusErrorWeight: 100,            // combined_error = angle_error + radius_error*100, line 288
    clusterSize: 1,                    // researcher decision D1 (source NUM_OF_IMAGES_PER_CLUSTER=64)
    clusterDistance: "euclidean2D",    // collect_nearest_images
    centerB: "negateA",                // researcher decision D3: B = −A (source opposite_point = -base_point)
    trainingSelection: "nearestUnused",// researcher decision D4 (source used_indices pattern)
    driftSelection: "nearestUnused",   // researcher decision D5: no repeats in drift
    exclusionScope: "perPhase",        // separate used sets: training excludes training, drift excludes drift only (D4, D5)
  },
  randomization: { seedPolicy: "crypto-random-uint32", algorithm: "sfc32", algorithmVersion: "1", trainingBlockSize: 4 /* placeholder OQ3 */ },
  ui: { feedbackDurationMs: 1000, imageDisplayScale: 2 },
  storage: { adapter: "localCsv", localStoragePrefix: "hx" },
  development: { enabled: false, allowUrlActivation: false },
};
```
- `validateConfig()` runs before consent. It checks: finite angles; `(initialAngleB − initialAngleA) mod 360 === 180` (D3); counts that are positive integers; `accuracyWindow ≤ minTrials` is not required (the condition `completed ≥ max(minTrials, window)` is handled in code); a threshold in [0,1]; an even `trainingBlockSize`; `clusterSize ≤ N`; supported enum values; and successful loading of the asset paths. On failure, an English error is shown and the experiment does not start.
- `resolveEffectiveConfig(base, devOverrides)` returns a frozen deep copy, and every override is recorded in `metadata.developmentOverrides`.
- No duplicate initial angles: `drift` holds no angles of its own and uses `training.initialAngleA/B`.

### Stage 3 — Scientific modules (pure, no DOM)
| Module | Function | Source counterpart |
| --- | --- | --- |
| `trajectory.js` | `angleDeg([x,y]) = ((atan2(y,x)·180/π) % 360 + 360) % 360` | ANGLE_MAP |
| | `rotateVector(v, deg)`: CCW matrix, same order of operations as numpy | `rotate_vector` |
| | `createBasePoint(points, angle, {targetRadius, radiusErrorWeight})` → `{index, point}` via argmin, with the first index on ties (as `np.argmin`) | `create_base_and_opposite_points` |
| | `initialCenters(cfg, pca)`: A = base point; B = `−A` (D3) | `opposite_point = -base_point` |
| `clusterConstruction.js` | `collectNearest(center, points, k, excluded?)` → indices sorted by Euclidean distance, ties broken by index (documented; numpy's tie order is undefined). Indices in `excluded` get distance ∞, as `dists[idx] = np.inf` in the source | `collect_nearest_images` + `used_indices` from `generate_rotation_sequence` |
| `stimulusSelector.js` | `selectStimulus({group, centers, pca, excluded, cfg})` → `{imageIndex, imageId, pc1, pc2, stimulusAngle, distanceToCenter}`. Training: `collectNearest(center, k=1, usedTrainingIndices)`; drift: `collectNearest(center, k=1, usedDriftIndices)`. A pure function, no RNG. If every image has been used, an explicit error is thrown (cannot happen with the defaults: 11,817 images) | D1, D2, D4, D5 |
| `pcaLoader.js` | Loads `pca.json` and the manifest, validates, returns `Float64Array`s | `load_top2_filtered` |

- Centers are held as **vectors** and rotated **incrementally** after every response, as in the source, not via a closed-form formula. The nominal angle is computed separately for recording.

### Stage 4 — Seeded randomization (`seededRandomization.js`)
- `sessionSeed` = a uint32 from `crypto.getRandomValues` (seed generation itself is documented), or a seed set in development mode.
- One documented PRNG (sfc32), with two **sub-streams** derived from the same seed through a fixed hash: `trainingSchedule` and `driftSchedule`. This way the number of training trials, which depends on responses, does not shift the drift sequence. There is no image-selection stream, because selection is deterministic (D2, D4). The experiment code contains no `Math.random()`, and a lint/grep check verifies this.
- **Drift**: a list with `floor(N/2)` A and `floor(N/2)` B; if N is odd, the extra group is chosen by the RNG. The list is shuffled with Fisher–Yates and generated in advance. The difference between the groups is ≤ 1.
- **Training**: blocks of `trainingBlockSize` (half A, half B), each block shuffled with Fisher–Yates. A new block is generated only when the previous one is exhausted.
- Metadata stores: the seed, algorithm+version, the effective drift schedule, the generated training blocks, and a draw counter per stream. Together with the responses and the PCA data, this is sufficient to fully reproduce the image sequence (including `usedTrainingIndices` and `usedDriftIndices`, which are derived from the records). The balancing policy is documented in the README.

### Stage 5 — Response coding and the training criterion
- `responseCoding.js`: `RESPONSES = ["A","Probably A","Probably B","B"]`; `toBinary()`; `isCorrect(binary, trueGroup)`.
- `trainingPhase.js`:
  ```js
  evaluateCriterion(history, {minTrials, accuracyWindow, accuracyThreshold}) {
    const n = history.length;
    const eligible = n >= minTrials && n >= accuracyWindow;
    if (!eligible) return { eligible, accuracy: null, passed: false };
    const last = history.slice(-accuracyWindow);
    const accuracy = last.filter(t => t.correct).length / accuracyWindow;
    return { eligible, accuracy, passed: accuracy >= accuracyThreshold };
  }
  ```
  The criterion is evaluated after **every** response. There is no cap and no timeout. The value is saved in the record (`windowAccuracy` as an additional field).

### Stage 6 — Experiment engine (`experimentEngine.js`)
State machine:
`init → (recovery?) → consent → instructions → training ⇄ feedback → transition → drift → completing → complete` (+ `error`, `devTerminated`).

**Trial loop (shared by both phases, in the exact order of PRD §7):**
1. `snapshot = {centerA, centerB, nominalAngleA, nominalAngleB}` — a copy, not a reference.
2. `group = schedule.next()`.
3. `stim = selectStimulus(group, snapshot)`: nearest among images not yet shown **in the same phase** (D4, D5).
4. `await presenter.show(stim.imageId)` → returns `onset` (see Stage 7). **On a load failure**: an English error with a Retry button is shown; the schedule does **not** advance (the group and stim are kept for the retry), no record is created and no rotation happens.
5. The first response is accepted → `rt = performance.now() − onset` (captured **first**), the buttons are disabled immediately and a record is created from the snapshot. `await storage.saveTrial(record)`; if saving fails, a visible error is shown and nothing advances. Only after a successful save is the image added to the current phase's used set (`usedTrainingIndices` or `usedDriftIndices`).
6. **Training**: `Correct`/`Incorrect` feedback for `feedbackDurationMs`, then criterion evaluation (on a trial that has already received feedback). If passed → `transition`.
   **Drift**: `centerA = rotateVector(centerA, step)`, `centerB = rotateVector(centerB, step)`, `driftIndex++`.
7. Next trial, or `completing` after `numberOfTrials` drift responses.

- In `transition`, drift begins only after `Continue` is clicked.
- Duplicate protection: an `acceptingResponse` flag at the engine level and `disabled` at the DOM level. The record key is `${sessionId}:${trialNumber}`, and `saveTrial` is idempotent (an existing key is not written again).
- The engine knows nothing about the DOM or localStorage; it receives `presenter`, `storage`, `clock` and `rngFactory` by injection. This enables tests with fakes.

### Stage 7 — Image presentation and timing (`stimulusPresenter.js`)
Documented onset procedure:
1. `const img = new Image(); img.src = url; await img.decode()`. Loading only; the timer is not running yet.
2. Replace the previous image in the DOM (the previous image is already hidden after the response, so a stale face is never shown).
3. `requestAnimationFrame(() => requestAnimationFrame(t => { onset = performance.now(); enableButtons(); }))`: the second rAF runs after the frame containing the image has been painted. Onset and enabling happen at the same moment.
4. Preloading the next trial: allowed only after its selection is fixed. In drift, only after the rotation; in training, only after the criterion evaluation. Preloading does not change the onset semantics.
5. A failed `onerror`/`decode()` → `ImageLoadError`, handled by the engine per Stage 6.4.

### Stage 8 — UI (English)
- **Consent**: text from `content/consent.js` with `[RESEARCHER: study title]`, `[RESEARCHER: IRB/approval number]` and similar placeholders, without guessing content. A checkbox with the exact PRD wording; `Continue` is disabled until it is checked. `consentGiven: true` and `consentTimestamp` are recorded. In development mode there is a "Bypass consent (dev)" button, which records `consentGiven: false, consentBypassed: true`.
- **Instructions**: one face at a time, group A/B, the four buttons, and feedback on the initial trials. **No** mention of movement. Text in `content/instructions.js`.
- **Trial**: one centered image with `A | Probably A | Probably B | B` below it in a fixed order. Neutral background, no counters, no group colors and no cues of any kind during drift.
- **Feedback** (training only): `Correct` / `Incorrect` as neutral text.
- **Transition**: the exact PRD §4 wording and a `Continue` button.
- **Completion**: "Thank you…", an automatic download, and a `Download data` button for manual download or retry.
- **Recovery**: if localStorage holds an `in_progress` session, show "A previous session was interrupted" with two buttons: `Download its data` (CSV marked incomplete) and `Start a new session`. The old session is not deleted.
- **Errors**: an English banner for load, storage and download failures.

### Stage 9 — Storage (PRD §13)
- `storageContract.js`: JSDoc for `startSession(meta)`, `saveTrial(trial)` and `completeSession(completionMeta)`. All are async and return `{ok:true}` or throw a `StorageError {code, message, cause}` (codes: `QUOTA_EXCEEDED`, `UNAVAILABLE`, `SERIALIZATION`, `DOWNLOAD_BLOCKED`). Plus `assertImplementsContract(adapter)`.
- `index.js`: `createStorage(config)` selects the adapter by `config.storage.adapter`. This is the only place that knows about implementations.
- `localCsvStorage.js`:
  - Keys: `hx:index` (list of sessionIds), `hx:session:<id>:meta`, and `hx:session:<id>:trials` (an array, rewritten on every trial; ~500 records ≈ 300 KB). There is also an in-memory copy.
  - `startSession`: creates the entries **without** overwriting earlier sessions.
  - `saveTrial`: duplicate check by key → memory → localStorage, and an update of `lastTrialNumber`, `phase` and `counts` in the meta. On failure: in-memory rollback and a `StorageError`. No CSV download at this point.
  - `completeSession`: writes the final meta and then downloads `<sessionId>.csv`. If the download is blocked, it returns `{ok:true, downloaded:false}` and the screen shows a manual button.
  - `exportSession(id, {incomplete})` and `listSessions()` serve the recovery screen and development. They are not part of the scientific contract.
- The README will include a prominent warning: **CSV and localStorage are for piloting only and are not a Prolific collection mechanism.**

### Stage 10 — CSV (`csvSerializer.js`, PRD §14)
- **Per-trial columns (fixed order):** `participantId, sessionId, trialNumber, phaseTrialNumber, phase, imageId, trueGroup, pc1, pc2, stimulusAngle, currentAngleA, currentAngleB, response, binaryResponse, correct, responseTimeMs, feedbackShown, timestamp`.
- **Additional reproduction columns:** `centerA_pc1, centerA_pc2, centerA_actualAngle, centerB_pc1, centerB_pc2, centerB_actualAngle, distanceToCenter, selectionRule` (`nearestUnused`/`nearest`), `windowAccuracy` (training).
- **Repeated metadata columns (on every row):** `experimentVersion, schemaVersion, participantIdSource, sessionStartTime, sessionEndTime, consentGiven, consentTimestamp, consentBypassed, developmentMode, sessionSeed, rngAlgorithm, pcaSha256, assetsManifestHash, trainingCompletedTrials, trainingCriterionMet, driftCompletedTrials, completionStatus, terminationReason, effectiveConfigJson, developmentOverridesJson, rngStateJson`.
- JSON serialized deterministically (sorted keys). RFC 4180 escaping (commas, quotes, newlines). Booleans `true`/`false`. Angles in degrees [0,360). RT in milliseconds. Timestamps in ISO 8601 UTC.
- File name: `<sessionId>.csv`, or `<sessionId>_INCOMPLETE.csv` for a partial export.

### Stage 11 — Identity and metadata (PRD §11)
- `sessionId = crypto.randomUUID()`.
- `participantId`: `PROLIFIC_PID` from the URL if present and non-empty (`participantIdSource: "prolific_url"`); otherwise `anon-<random>` (`"generated_anonymous"`). The full URL is not stored.
- Metadata includes every field in PRD §11, `completionStatus` ∈ {`in_progress`, `complete`, `incomplete_dev_early_finish`, `interrupted`} and `terminationReason`.

### Stage 12 — Development mode (PRD §15)
- Enabled only if `config.development.enabled === true`, or when running via `npm run dev` together with `?dev=1` (`import.meta.env.DEV`). In the participant build, the panel code is loaded only via dynamic import, so it does not appear in the UI.
- The panel includes: consent bypass; overrides for `minTrials` and `accuracyWindow` (both); `accuracyThreshold`, `numberOfTrials`, `degreesPerTrial` and the seed; a debug display (phase, trial #, group, nominal and actual angles, imageId, accuracy window); "Finish early", which records `completionStatus: incomplete_dev_early_finish` and `trainingCriterionMet` with its true value, not `true`; and a partial CSV download at any point once data exists.
- Every override goes into `developmentOverrides` and `effectiveConfig`.

### Stage 13 — Researcher visualization (PRD §Scientific visualization)
- An SVG implementation (`pcaPlot.js`) that reproduces the conventions of `create_prediction_scatter` and `plot_clusters_with_given_indices` (a documented copy of the parameters into the file, not an import): all points in gray (alpha 0.3); trajectory points (OQ2); a dashed circle of radius `max(r)·1.05`; radial lines and labels every 20°; axes; grid; 1:1 aspect ratio; PC1/PC2.
- Trajectory points (OQ2): the union of the images nearest to centers A and B at every drift position per the effective configuration, without exclusion (251 images with the defaults). The images actually shown come from the records.
- Layers: the current A (blue ×) and B (red *) centers; all shown stimuli (A blue, B red, filled); the **current** stimulus highlighted (thick ring / larger); a circle at the center radius.
- Data comes **from the actual trial records** (localStorage), not from a sample.
- Availability:
  1. `review.html` — selects a session from `hx:index` or loads an exported CSV; updates live via the `storage` event while the experiment runs in another tab.
  2. Inside the development panel (development mode only).
- Filters: phase (training/drift/all) and group (A/B/all), plus hover/click showing imageId, angle, trial #, response and a thumbnail.
- Export: `Export SVG` and `Export PNG` (canvas, 2×), with file name `<sessionId>_pca_plot.(svg|png)`. Files are downloaded by the browser; all visualization code lives under `human_experiment/`.
- `review.html` is not linked from the participant page.

### Stage 14 — Tests (mapped to the PRD §17 acceptance checks)

| # | Check | How |
| --- | --- | --- |
| 1 | Repository unchanged | `npm run verify:repo` compares against the baseline |
| 2 | Scientific parity | `make_parity_fixtures.py` (a copy of `create_base_and_opposite_points`, `rotate_vector`, `collect_nearest_images` and `used_indices`, on the real CSV) produces fixtures: base point and `−A` for 0°, 37°, 271.5°; a nearest-unused sequence of 60 training trials per group, followed by 360 drift trials (fixed group order from the fixture) at 1° and 0.5° steps, with a separate drift `used` set that starts empty. Vitest compares (positions ≤1e-12, identical indices) and checks near-ties |
| 2b | Researcher decisions | k=1; every `imageId` is unique within training and every `imageId` is unique within drift; a training image **may** appear in drift, and the drift sequence is identical regardless of training length; an image that failed to load is not counted as used; B = −A; a configuration with a separation other than 180° is rejected |
| 3 | One face and four buttons | jsdom: `querySelectorAll('img').length===1`, button text order, `response` saved verbatim |
| 4 | Training criterion | Unit: 19 correct → does not pass; 16/20 passes; 15/20 fails; 40 trials with 30 early correct and a final window of 15/20 → does not pass; centers fixed throughout training |
| 5 | Feedback, then transition | Integration: the deciding trial receives feedback → `transition` → no drift trial until `continue()` |
| 6 | Drift progression | Integration with fakes: trial 1 = 0°/180°, 2 = 1°/181°, 180 = 179°/359°, 181 = 180°/0°, 360 = 359°/179°; exactly 360 records; internal centers return to 0/180; independent angles (e.g. 30°/210°) and alternate steps (100×0.5°) |
| 7 | No feedback in drift | `feedbackShown=false`, `correct` computed, the DOM has no feedback text or color |
| 8 | Timing and failures | A fake presenter with a 2 s load delay → RT excludes the delay; load failure → no record and no rotation; two clicks → one record |
| 9 | Reproducibility | Same seed + same responses → same groups and imageIds; drift balance ≤1 and balance within every training block |
| 10 | Storage and recovery | Fake localStorage: save, "refresh" (new engine), recovery and export; quota error → visible error |
| 11 | CSV | Re-parse: one row per response, all columns, escaping, partial export marked |
| 12 | Identity and consent | With and without `PROLIFIC_PID`; disabled button; bypass recorded |
| 13 | No dev in participant mode; replaceable adapter | `build` + grep of dist; a test with a `memoryStorage` that implements the contract without changing the engine |
| — | Playwright (smoke) | A full run with short overrides in a real browser, including the CSV download |

### Stage 15 — README and handoff (PRD §17)
The README will include: installation, asset preparation, run and build instructions; a configuration table; the trial and session schema with units; a provenance table (source file and lines ↔ copied module); the Section 1 findings and the Section 2 open questions with the decisions made; the onset procedure; the balancing policy; acceptance check results; and an explicit warning that this version is **not ready for Prolific collection** without a server-backed adapter.

---

## 6. Recommended execution order and checkpoints

1. Stages 0–1. The main scientific decisions (D1 to D6) have been made; the questions in Section 2.3 are implemented with the marked defaults and can be changed in the configuration only.
2. Stages 2–5 with unit and parity tests.
3. Stages 6–9 with integration tests.
4. Stages 10–12.
5. Stage 13.
6. Stages 14–15 and a final `verify:repo`.
