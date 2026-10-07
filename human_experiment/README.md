# Human Experiment — Representation Stability Under Drifting Stimuli

A browser experiment for human participants, parallel to the neural-network experiment in this repository. Participants learn to classify faces into Group A or Group B with feedback. They then keep classifying without feedback while the two category centers rotate along the PCA trajectory used by the network experiment.

Specification: [HUMAN_EXPERIMENT_PRD.md](HUMAN_EXPERIMENT_PRD.md). Build plan and researcher decisions: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

> **⚠ Pilot-only data storage.** This version saves data in the participant's browser (`localStorage`) and downloads a CSV file to the participant's computer. **It is NOT a data collection mechanism for Prolific**: a browser download stays on the participant's machine and never reaches the researcher. Online collection requires a server-backed storage adapter (see [Storage](#storage)), which is not implemented here. Do not run this version on Prolific.

---

## Quick start

All commands run inside `human_experiment/`. Requirements: Node.js ≥ 20. Python 3 with numpy and pandas is needed only to regenerate the parity fixtures. Google Chrome is needed only for the end-to-end tests.

```bash
npm install                 # dev dependencies (vite, vitest, jsdom, @playwright/test)
npm run prepare-assets      # copy faces + derive PCA JSON into public/assets/ (read-only on the source)
npm run dev                 # http://localhost:5173/            participant view
                            # http://localhost:5173/?dev=1      development mode
                            # http://localhost:5173/review.html researcher review page
npm run build               # production build -> dist/ (static files, ~105 MB incl. faces)
npm run preview             # serve dist/ on http://localhost:4173/
```

To serve the study, deploy the contents of `dist/` to any static web server. Participant links can carry `?PROLIFIC_PID=...`.

| Script | Purpose |
| --- | --- |
| `npm run prepare-assets` | Reads `../pca_top2_filtered_female_vgg_1.csv` and `../female_faces/` (read-only). Writes `public/assets/faces/`, `public/assets/pca.json` and `public/assets/assets-manifest.json`. Fails loudly if an image is missing. |
| `npm test` | Unit and integration tests (Vitest, jsdom). Requires `prepare-assets` first. |
| `npm run test:e2e` | Real-browser tests in the installed Chrome (Playwright). Run `npm run build` first. |
| `npm run parity:fixtures` | Regenerates `tests/fixtures/parity/parity.json` from verbatim copies of the source Python functions. |
| `npm run verify:repo` | Confirms that no pre-existing repository file changed and that all new files are under `human_experiment/`. |
| `npm run verify:participant-build` | Confirms that development mode is disabled and lazily loaded in `dist/`. |

---

## Participant flow

`Consent → Instructions → Training with feedback → Transition → Drift without feedback → Completion and CSV download`

- **Consent**: the study text lives in [src/content/consent.js](src/content/consent.js) and consists entirely of `[RESEARCHER: ...]` placeholders, which must be replaced with approved text. The checkbox uses the exact PRD wording. `Continue` stays disabled until the box is checked.
- **Instructions** ([src/content/instructions.js](src/content/instructions.js)): explain one face at a time, the A/B groups, the four answers, and initial feedback. They do not mention the drift.
- **Trial**: one centered face, with the buttons `A / Probably A / Probably B / B` below it in a fixed order. Buttons are enabled only at image onset and disabled on the first accepted click.
- **Training**: `Correct` / `Incorrect` feedback (1 s) after every answer, centers fixed. Training ends only when ≥ 20 trials are completed **and** the last 20 trials are ≥ 80% correct (sliding window). The qualifying trial also receives feedback.
- **Transition**: the exact PRD text. Drift starts only after `Continue`.
- **Drift**: 360 trials. No feedback, no cues. Both centers rotate 1° counterclockwise **after** each response.
- **Completion**: `<sessionId>.csv` downloads automatically. A `Download data` button retries the download.
- **Recovery**: when the page is reloaded during a session, the interrupted session is kept and offered for export (`<sessionId>_INCOMPLETE.csv`). A new session never overwrites it.

All participant-facing text is in [src/content/](src/content/) and can be edited without touching the logic.

---

## Scientific logic and provenance

### Inspected source

| Source (read-only) | What was taken | Copy / port in `human_experiment/` |
| --- | --- | --- |
| `network_classification/classify_rotation_resnet50.py` | Angle convention (lines 52-56), `load_top2_filtered` (236-251), `create_base_and_opposite_points` (255-297), `rotate_vector` (300-318), `collect_nearest_images` (321-357), `generate_rotation_sequence` (760-804), the main loop's order "select at current centers, then rotate" (1512-1784), and plot conventions of `create_prediction_scatter` (807-907) | Verbatim file copy: [reference_source/classify_rotation_resnet50.py](reference_source/classify_rotation_resnet50.py). Verbatim function copies: [scripts/reference/source_functions.py](scripts/reference/source_functions.py). JS ports: [src/scientific/](src/scientific/), [src/experiment/driftPhase.js](src/experiment/driftPhase.js), [src/visualization/pcaPlot.js](src/visualization/pcaPlot.js) |
| `network_classification/tools/generate_rotation_sequence.py` | `plot_clusters_with_given_indices` conventions (205-259): base × and opposite * markers | Verbatim copy: [reference_source/generate_rotation_sequence.py](reference_source/generate_rotation_sequence.py). Port: [src/visualization/pcaPlot.js](src/visualization/pcaPlot.js) |
| `extract_embeddings/PCA_2_and_rest.py` | How the PCA file was produced (FaceNet/InceptionResNetV1 embeddings, top-2 PCs, 10% smallest residual radius) | Verbatim copy: [reference_source/PCA_2_and_rest.py](reference_source/PCA_2_and_rest.py) (documentation only) |
| `pca_top2_filtered_female_vgg_1.csv` | 11,817 rows `filename, PC1, PC2`, no header | Derived: `public/assets/pca.json` (same row order; SHA-256 recorded) |
| `female_faces/` | JPEG 178×218 images | Copied: `public/assets/faces/` (gitignored, regenerated by `prepare-assets`) |

### Source-derived conventions

- **Coordinates**: PC1 = x, PC2 = y, origin (0, 0).
- **Angles**: `degrees(atan2(PC2, PC1)) mod 360`, in degrees, range [0, 360).
- **Rotation**: counterclockwise rotation matrix (`rotate_vector`), applied **incrementally** to the center vectors after each drift response.
- **Center A**: the actual data point minimizing `angle_error + 100·|r − 0.45|` for the configured angle. For 0° this is `183785.jpg` at 359.49°, r = 0.4445.
- **Center B**: `−A`, the reflection through the origin (`opposite_point = -base_point`).

### Researcher decisions (IMPLEMENTATION_PLAN.md §2.1)

| # | Decision | Implementation |
| --- | --- | --- |
| D1 | Cluster size 1 (the source uses 64) | `scientific.clusterSize: 1`, same nearest-neighbor function |
| D2 | One face per trial = the single image in the size-1 cluster | Deterministic: nearest image to the selected group's current center |
| D3 | B is always 180° from A | `B = −A`. Configurations whose angles are not 180° apart are rejected |
| D4 | Training: nearest image not yet selected in training | `usedTrainingIndices` |
| D5 | Drift: no repeats within drift; training images may reappear | Separate `usedDriftIndices`, empty at drift start |
| D6 | PCA file is `pca_top2_filtered_female_vgg_1.csv` | — |

An image counts as "used" only after its response has been saved. A failed image load retries the same image.

### Open questions implemented with marked defaults (IMPLEMENTATION_PLAN.md §2.3)

| # | Default |
| --- | --- |
| OQ1 | `currentAngleA/B` hold the nominal angle (`initial + (k−1)·step`). The actual center angle is in `centerA_actualAngle` / `centerB_actualAngle`, which sit 0.51° below nominal with the default data. |
| OQ2 | Visualization "trajectory candidates" = images nearest to A or B at any drift position, without exclusion (251 images by default). The source `generate_rotation_sequence` is an optional layer. |
| OQ3 | `feedbackDurationMs: 1000`, `trainingBlockSize: 4`, display scale ×2 (356×436 px). |

### Measured consequences (default configuration)

- The first A training trial shows the center image itself (distance 0).
- **Training**: shown images drift away from the fixed centers as the nearest ones are used up. For A, the distance is 0.037 at the group's 20th trial and 0.060 at its 60th.
- **Drift**: A and B pass over the same ring, so B often gets the next-nearest image after A used the nearest one. Distance to center: median 0.006, 90th percentile 0.058, maximum 0.104.
- The actual distance is recorded per trial in `distanceToCenter`.

---

## Configuration

Single entry point: [src/config/experimentConfig.js](src/config/experimentConfig.js). The configuration is validated before the experiment starts ([src/config/validateConfig.js](src/config/validateConfig.js)). The effective configuration, including development overrides, is stored in every session's metadata.

| Key | Default | Notes |
| --- | --- | --- |
| `training.initialAngleA` / `initialAngleB` | 0 / 180 | Also the drift start. Must be 180° apart (D3). |
| `training.minTrials` / `accuracyWindow` / `accuracyThreshold` | 20 / 20 / 0.8 | Sliding-window criterion. No cap or timeout. |
| `drift.numberOfTrials` / `degreesPerTrial` | 360 / 1 | Independent of each other. |
| `scientific.*` | see file | Source-derived (`targetRadius 0.45`, `radiusErrorWeight 100`, CCW, Euclidean) plus decisions D1–D5. |
| `randomization.trainingBlockSize` | 4 | Placeholder OQ3. Must be even. |
| `ui.feedbackDurationMs`, `ui.imageDisplayScale` | 1000, 2 | Placeholders OQ3. |
| `storage.adapter` | `localCsv` | `memory` is also available (tests). |
| `development.enabled` | `false` | See [Development mode](#development-mode). |

### Randomization and reproducibility

- **Seed**: each session gets a `sessionSeed` (uint32 from `crypto.getRandomValues`), or a seed set in development mode.
- **Generator**: one documented generator, sfc32 v1, with two independent named streams seeded from `(sessionSeed, name)` via FNV-1a and splitmix32.
  - `trainingSchedule`: balanced shuffled blocks of `trainingBlockSize` (2 A + 2 B), generated on demand.
  - `driftSchedule`: a single shuffled list of ⌊N/2⌋ A + ⌊N/2⌋ B, plus one random extra group when N is odd. |A − B| ≤ 1.
- **Image selection**: deterministic (no randomness).
- **Independence from training length**: because the streams are separate, the drift sequence is the same however long training takes.
- **Stored for reproduction**: seed, algorithm and version, the drift schedule, the generated training blocks, draw counts, the effective configuration, and the PCA SHA-256 and face-list hash.
- **Reproduction**: the same seed, data, configuration, and response history reproduce every group and image. `Math.random()` is not used anywhere (checked by a test).

---

## Timing

Response time is measured with `performance.now()` ([src/components/trialScreen.js](src/components/trialScreen.js)):

1. Responses are disabled and the previous face is removed.
2. The next image is fetched and fully decoded off-screen (`await img.decode()`). No timer runs during loading.
3. The decoded image is inserted into the empty frame.
4. In the second `requestAnimationFrame` callback, after the frame with the image has been rendered, `onset = performance.now()` is taken and the buttons are enabled **at the same moment**.
5. On click, `performance.now()` is captured first. `responseTimeMs = click − onset`, computed before saving or feedback.

As a result, image loading, feedback duration and storage latency never enter the response time. A failed load shows an English error with `Try again`. It creates no record, starts no timer and does not move the centers. Duplicate clicks are ignored: only the first click per trial is accepted, and the record key `sessionId:trialNumber` is idempotent.

---

## Data

### Trial record and CSV columns

`<sessionId>.csv` has one row per answered trial and a fixed header. Booleans are written as `true`/`false`, angles in degrees [0, 360), times in milliseconds, and timestamps in ISO 8601 UTC. Fields follow RFC 4180 quoting.

| Column | Meaning |
| --- | --- |
| `participantId`, `sessionId` | `PROLIFIC_PID` URL parameter if present, else `anon-<16 hex>`; session UUID |
| `trialNumber`, `phaseTrialNumber`, `phase` | One-based overall index; one-based index within `training` or `drift` |
| `imageId`, `imageIndex` | Source file name; row index in the PCA file |
| `trueGroup` | Group (A/B) whose center selected the image |
| `pc1`, `pc2`, `stimulusAngle` | Actual PCA coordinates and angle of the shown image |
| `currentAngleA`, `currentAngleB` | Nominal center angles that generated this display (snapshot taken before any rotation) |
| `response`, `binaryResponse`, `correct` | Original four-level answer, verbatim; derived A/B; `binaryResponse === trueGroup` (also recorded in drift) |
| `responseTimeMs` | Image onset → accepted click |
| `feedbackShown`, `timestamp` | `true` in training only; wall-clock time of the accepted response |
| `distanceToCenter`, `selectionRule` | Distance from the image to its group center; `nearestUnused` |
| `centerA_pc1`, `centerA_pc2`, `centerA_actualAngle`, `centerB_*` | Exact center vectors and actual angles used for this display |
| `windowAccuracy`, `criterionPassed` | Training only: last-window accuracy after this trial; criterion outcome |
| `imageLoadFailures` | Failed load attempts before this trial's image displayed |

**Session metadata** (repeated on every row): `experimentVersion`, `schemaVersion`, `participantIdSource`, `sessionStartTime`, `sessionEndTime`, `consentGiven`, `consentTimestamp`, `consentBypassed`, `developmentMode`, `sessionSeed`, `rngAlgorithm`, `rngAlgorithmVersion`, `pcaSource`, `pcaSha256`, `facesListSha256`, `trainingCompletedTrials`, `trainingCriterionMet`, `trainingCriterionMetAtTrial`, `driftCompletedTrials`, `completionStatus` (`complete`, `in_progress`, `interrupted` or `incomplete_dev_early_finish`), `terminationReason`, `exportedAsIncomplete`, and the deterministic (key-sorted) JSON columns `initialCentersJson`, `effectiveConfigJson`, `developmentOverridesJson` and `randomizationJson`. Counts and criterion outcome are derived from the saved records, so they are also correct for interrupted sessions.

### Storage

Phase logic depends only on the contract in [src/services/dataStorage/storageContract.js](src/services/dataStorage/storageContract.js):

```js
await storage.startSession(sessionMetadata);
await storage.saveTrial(trialData);
await storage.completeSession(completionMetadata);
```

Failures reject with `StorageError` (`QUOTA_EXCEEDED`, `UNAVAILABLE`, `SERIALIZATION`, `NOT_FOUND`, `INVALID`). The engine shows them in English and does not advance; `Try again` re-saves the same record.

The pilot adapter, [localCsvStorage.js](src/services/dataStorage/localCsvStorage.js), keeps the session in memory and in `localStorage`. Each session has its own keys:
- `hx:index`
- `hx:session:<id>:meta`
- `hx:session:<id>:trials`

It never overwrites another session, never downloads a CSV per trial, and downloads the CSV at completion. Saved data stays in `localStorage` after export.

**Adding a server or Supabase adapter**:
1. Implement the three methods (plus optional `exportSession`/`listSessions`/`getSession`/`markInterrupted`).
2. Register it in [src/services/dataStorage/index.js](src/services/dataStorage/index.js).
3. Set `storage.adapter`.

No phase, selection, timing or response-coding code changes. The engine tests run unchanged on both the memory and the localStorage adapters.

---

## Development mode

Development mode is off by default and opens in one of two ways:
- `npm run dev` with `?dev=1`. Production builds ignore `?dev=1`.
- Setting `development.enabled: true` in the configuration.

Its code is loaded lazily and never appears in participant mode.

- **Setup screen**: overrides for minimum training trials, accuracy window, threshold, drift trial count, degrees per trial, and session seed. Overrides are validated and recorded in `developmentOverrides` and `effectiveConfig`.
- **Consent bypass**: recorded as `consentBypassed: true, consentGiven: false`.
- **Debug panel** (docked right): phase, trial numbers, true group, image, stimulus angle, nominal and actual center angles, window accuracy, seed.
- **Finish early**: ends the session as `incomplete_dev_early_finish`. The criterion is never marked as met unless it actually was.
- **Download partial CSV**: available at any point after data exists; the file is marked `_INCOMPLETE`.
- **PCA plot**: the live researcher visualization (below).

## Researcher visualization

Available in the development panel and at `review.html`. The review page is not linked from the participant page and warns that it must not be shown to participants. It reproduces the source plot conventions:
- all images in gray (alpha 0.3)
- a dashed reference circle at `max(r)·1.05`
- 20° spokes with labels
- x/y axes, grid, equal aspect ratio, PC1/PC2 labels
- A in blue and B in red
- the A center as a black ×, the B center as a black *

It adds:
- the circular trajectory of the centers (dotted)
- trajectory candidates (OQ2)
- every presented stimulus, taken from the actual trial records (training points have a white outline)
- the current or most recent stimulus, ringed

Filters cover phase (all/training/drift) and group (all/A/B), and an optional source rotation-sequence layer can be turned on. Clicking a point shows its details and a thumbnail. **Export SVG** and **Export PNG** save `<sessionId>_pca_plot.svg|png`.

The review page reads sessions from this browser's `localStorage` and updates live while the experiment runs in another tab. It can also load an exported CSV.

---

## Verification results

Results as of 2026-10-07: `npm test` 72/72 passed, `npm run test:e2e` 2/2 passed (Chrome), `verify:repo` OK, `verify:participant-build` OK.

| PRD §17 check | Evidence |
| --- | --- |
| 1. No pre-existing file modified; output confined to `human_experiment/` | `npm run verify:repo` checks against a baseline taken before work (HEAD 48cadce): no tracked file outside `human_experiment/` changed; all untracked files are inside it; ignored files outside are unchanged (size and mtime); PCA SHA-256 unchanged. **One reported exception:** `human_experiment/IMPLEMENTATION_PLAN.md`, which was committed before work began, was translated to English at the user's explicit request (listed in `scripts/repo-baseline.json → userAuthorizedChanges`). |
| 2. Trajectory, clusters, rotation and selection match the source | [tests/unit/parity.test.js](tests/unit/parity.test.js) runs against fixtures generated by the source functions on the real CSV: identical base and opposite points (0°, 37°, 180°, 271.5°), identical k = 1/5/64 clusters, rotations within 1e-12, and identical training (120 trials) and drift (360 trials at 1°, 200 at 0.5°) selection sequences. No near-ties exist that could make the order platform-dependent. |
| 3. One face, fixed buttons, verbatim responses | [tests/integration/ui.test.js](tests/integration/ui.test.js) and e2e: exactly one `<img>`, button order, responses stored verbatim. |
| 4. Training criterion | 19 correct does not pass; 16/20 passes; 15/20 fails; a high cumulative score with a failing last-20 window does not pass; centers fixed. ([engine.test.js](tests/integration/engine.test.js), [logic.test.js](tests/unit/logic.test.js)) |
| 5. Feedback, then transition; drift waits | The qualifying trial gets feedback, then the transition; no drift trial before `Continue`. |
| 6. Drift progression | Trials 1/2/180/181/360 at 0/180, 1/181, 179/359, 180/0, 359/179; exactly 360 records; centers return to start; alternate centers (30/210) and steps (100 × 0.5°). |
| 7. No feedback in drift | `feedbackShown=false`, `correct` recorded, no feedback text in the DOM. |
| 8. Timing and failures | A 1.5–2 s load delay does not change RT; a failed load creates no record and no rotation and retries the same image; double clicks give one record. |
| 9. Reproducibility and balancing | Same seed + responses give identical sequences; different seeds differ; drift balance ≤ 1; every training block is balanced; drift is independent of training length. |
| 10. Persistence and recovery | localStorage save; recovery after a reload in real Chrome; quota and unavailable errors surfaced; sessions never overwritten. |
| 11. CSV | One row per response, every column, RFC 4180 escaping, metadata columns, `_INCOMPLETE` partial exports, manual re-download. |
| 12. Identity and consent | `PROLIFIC_PID` used when non-empty, else anonymous; `Continue` disabled until consent; bypass only in development mode and recorded honestly. |
| 13. No dev in participant mode; replaceable adapter | e2e on the production build with `?dev=1` shows no development controls; `verify:participant-build`; engine runs unchanged on the memory and localStorage adapters. |

---

## Known limitations

- **Pilot-only storage**: see the warning at the top.
- **Placeholder consent text**: the consent text and the Prolific completion step are placeholders and must be completed by the researcher.
- **Onset precision**: onset is tied to the browser's rendering frame (about 16 ms resolution at 60 Hz). Actual display latency of the monitor is not measurable from the browser.
- **Size**: `public/assets/faces/` (85.7 MB, 11,817 images) is regenerated by `prepare-assets` and is not tracked. `public/assets/pca.json` and the manifest are small derived files.

## Directory layout

```text
human_experiment/
├── index.html, review.html          participant page, researcher review page
├── src/
│   ├── main.js, app.js, review.js   composition roots and screen rendering
│   ├── config/                      experimentConfig.js (single entry point), validateConfig.js
│   ├── content/                     researcher-editable English text
│   ├── components/                  screens, trial screen / presenter
│   ├── experiment/                  engine, training, drift, response coding, randomization, session
│   ├── scientific/                  PCA loader, trajectory, cluster construction, stimulus selector
│   ├── services/dataStorage/        contract, adapters, CSV serializer
│   ├── visualization/               PCA plot, viewer, export, trajectory membership
│   └── dev/                         development setup screen and panel (lazy)
├── scripts/                         prepare-assets, verify scripts, reference/ (Python parity)
├── reference_source/                verbatim copies of the inspected source files
├── public/assets/                   derived PCA JSON, manifest, copied faces
└── tests/                           unit, integration (jsdom), e2e (Playwright), parity fixtures
```
