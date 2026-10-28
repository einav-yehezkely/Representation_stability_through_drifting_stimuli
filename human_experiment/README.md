# Human Experiment — Representation Stability Under Drifting Stimuli

A browser experiment for human participants, parallel to the neural-network experiment in this repository. Participants learn to classify faces into Group A or Group B with feedback. They then keep classifying without feedback while the two category centers rotate along the PCA trajectory used by the network experiment.

Specification: [HUMAN_EXPERIMENT_PRD.md](HUMAN_EXPERIMENT_PRD.md). Build plan and researcher decisions: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

> **Data storage.** The production build sends every answer to `save.php` on the web server, which writes one file per session outside the web root (see [Storage](#storage) and [Deployment](#deployment-opalstack)). The Vite dev server (`npm run dev`) cannot run PHP, so it keeps the data in this browser's `localStorage`. Participants are never offered a download of their data; the researcher exports it from the [admin page](#admin-page). Before running on Prolific, replace the consent placeholders and set `completion.prolificCompletionUrl`.

---

## Quick start

All commands run inside `human_experiment/`. Requirements: Node.js ≥ 20. Python 3 with numpy and pandas is needed only to regenerate the parity fixtures. Google Chrome is needed only for the end-to-end tests.

```bash
npm install                 # dev dependencies (vite, vitest, jsdom, @playwright/test)
npm run prepare-assets      # copy faces + derive PCA JSON into public/assets/ (read-only on the source)
npm run dev                 # http://localhost:5173/            participant view
                            # http://localhost:5173/?dev=1      development mode
                            # http://localhost:5173/review.html researcher review page
                            # http://localhost:5173/admin.html  admin page (sessions saved in this browser)
npm run build               # production build -> dist/ (static files, ~105 MB incl. faces)
npm run preview             # serve dist/ on http://localhost:4173/
```

To serve the study, deploy the contents of `dist/` to a web server with PHP (see [Deployment](#deployment-opalstack)). Participant links can carry `?PROLIFIC_PID=...`.

| Script | Purpose |
| --- | --- |
| `npm run prepare-assets` | Reads `../pca_top2_filtered_female_vgg_1.csv` and `../female_faces/` (read-only). Writes `public/assets/faces/`, `public/assets/pca.json` and `public/assets/assets-manifest.json`. Fails loudly if an image is missing. |
| `npm test` | Unit and integration tests (Vitest, jsdom). Requires `prepare-assets` first. |
| `npm run test:e2e` | Real-browser tests in the installed Chrome (Playwright). Run `npm run build` first. |
| `npm run parity:fixtures` | Regenerates `tests/fixtures/parity/parity.json` from verbatim copies of the source Python functions. |
| `npm run verify:repo` | Confirms that no pre-existing repository file changed and that all new files are under `human_experiment/`. |
| `npm run verify:participant-build` | Confirms that development mode is disabled and lazily loaded in `dist/`. |
| `npm run data:csv -- <folder>` | Converts the server's `*.jsonl` session files to CSV (one file per session plus `all_sessions.csv`) in `<folder>/csv/`. |

---

## Participant flow

`Consent → Instructions → Training with feedback → Transition → Drift without feedback → Completion`

- **Consent**: the study text lives in [src/content/consent.js](src/content/consent.js) and consists entirely of `[RESEARCHER: ...]` placeholders, which must be replaced with approved text. The checkbox uses the exact PRD wording. `Continue` stays disabled until the box is checked.
- **Instructions** ([src/content/instructions.js](src/content/instructions.js)): explain one face at a time, the A/B groups, the four answers, and initial feedback. They do not mention the drift.
- **Trial**: one centered face, with the buttons `A / Probably A / Probably B / B` below it in a fixed order. Buttons are enabled only at image onset and disabled on the first accepted click.
- **Training**: `Correct` / `Incorrect` feedback (1 s) after every answer, centers fixed. Training ends only when ≥ 20 trials are completed **and** the last 20 trials are ≥ 80% correct (sliding window). The qualifying trial also receives feedback.
- **Transition**: the exact PRD text. Drift starts only after `Continue`.
- **Drift**: 360 trials. No feedback, no cues. Both centers rotate 1° counterclockwise **after** each response.
- **Completion**: "Your responses have been saved" and, when `completion.prolificCompletionUrl` is set, a `Return to Prolific` button. Participants are never offered a download of their data, with either storage adapter.
- **Recovery**: when the page is reloaded during a session, the interrupted session keeps its saved data, is marked `interrupted` (on the server, and in this browser), and a new session starts; neither overwrites the other. No download is offered.

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
| OQ2 | Visualization "trajectory candidates" = images nearest to A or B at any drift position, without exclusion (251 images by default). |
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
| `training.initialAngleA` / `initialAngleB` | 0 / 180 | Also the drift start. Must be 180° apart (D3). Default only: the admin page can change it (see [Experiment settings](#experiment-settings)). |
| `training.minTrials` / `accuracyWindow` / `accuracyThreshold` | 20 / 20 / 0.8 | Sliding-window criterion. No cap or timeout. |
| `drift.numberOfTrials` / `degreesPerTrial` | 360 / 1 | Independent of each other. Defaults only: the admin page can change them (see [Experiment settings](#experiment-settings)). |
| `scientific.*` | see file | Source-derived (`targetRadius 0.45`, `radiusErrorWeight 100`, CCW, Euclidean) plus decisions D1–D5. |
| `randomization.seedPolicy`, `randomization.fixedSeed` | `"fixed"`, 20261014 | One seed for all sessions (uint32). See [Randomization](#randomization-and-reproducibility). |
| `randomization.trainingBlockSize` | 4 | Placeholder OQ3. Must be even. |
| `ui.feedbackDurationMs`, `ui.imageDisplayScale` | 1000, 2 | Placeholders OQ3. |
| `ui.preloadCount` | 3 | Upcoming images downloaded in advance (performance only; 0 disables). See [Timing](#timing). |
| `storage.adapter` | `server` | `localCsv` (pilot) and `memory` (tests) are also available. |
| `storage.developmentAdapter` | `localCsv` | Used by the Vite dev server, which cannot run PHP. |
| `storage.serverEndpoint`, `requestTimeoutMs`, `maxAttempts` | `save.php`, 15000, 3 | Attempts per save before the participant sees `Try again`. |
| `completion.prolificCompletionUrl` | `""` | Prolific completion link (`https://...?cc=...`). Empty: no link is shown. |
| `development.enabled` | `false` | See [Development mode](#development-mode). |

### Randomization and reproducibility

- **Seed**: every session uses the same `sessionSeed`, `randomization.fixedSeed` (default 20261014), so every participant sees the same group order and the same faces in the same order. It can only be changed in [src/config/experimentConfig.js](src/config/experimentConfig.js); development mode and the admin page show it but cannot change it. The seed is still recorded in every session.
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

**Preloading.** Image selection is deterministic and the group schedules depend only on the seed, so the engine knows the next faces in advance ([src/experiment/sequencePlanner.js](src/experiment/sequencePlanner.js)). It announces the next `ui.preloadCount` images (default 3) at the instructions screen and right after each accepted answer, while that answer is being saved. The trial screen downloads and decodes them off-screen. Step 2 then finds the image already decoded, so the next face appears as soon as the save is confirmed, without waiting for an image download. Preloading changes neither which face is shown nor when the timer starts: onset is still taken after the frame with the new face is painted. The lookahead works on copies of the state (`TrainingSchedule.peek`, `DriftState.clone`), and a test checks that sessions are identical with and without it. A failed preload falls back to a normal load. Measured on the production build with a simulated 120 ms save and 80 ms image latency, the time from click to the next face fell from 252 ms to 167 ms; the remainder is the save round trip, which is kept so that a face never advances before its answer is stored.

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

It never overwrites another session and never downloads anything by itself: `completeSession()` only saves the final metadata. `exportSession()` writes a CSV on request (development panel, admin page under `npm run dev`); the data stays in `localStorage` afterwards.

The server adapter, [serverStorage.js](src/services/dataStorage/serverStorage.js), POSTs each call to [public/save.php](public/save.php) and resolves only after the server confirmed the write. Network and 5xx errors are retried (`maxAttempts`); a rejected request (4xx) is not. It also keeps a best-effort copy in `localStorage` through the pilot adapter; a failing copy never blocks the participant. `save.php` appends to `~/exp_data/representation_stability/<sessionId>.jsonl`, one JSON object per line (`start`, `trial`, `complete`, `interrupted`). Trials are idempotent on `sessionId:trialNumber`. [serverRecords.js](src/services/dataStorage/serverRecords.js) reads these files back and `npm run data:csv` writes them with the same CSV serializer, so the columns are identical to the pilot CSV. A session without a `complete` record is exported as `interrupted` and `_INCOMPLETE`.

**Adding another adapter** (e.g. Supabase):
1. Implement the three methods (plus optional `exportSession`/`listSessions`/`getSession`/`markInterrupted`).
2. Register it in [src/services/dataStorage/index.js](src/services/dataStorage/index.js).
3. Set `storage.adapter`.

No phase, selection, timing or response-coding code changes. The engine tests run unchanged on the memory, localStorage and server adapters.

---

## Deployment (Opalstack)

One-time setup:
1. Opalstack dashboard: create an Application of type **PHP-FPM + Apache** under the OS user used for SFTP, and a Site that routes a path of the domain to it, with Let's Encrypt enabled.
2. Over SSH, create the data folder next to `apps/` (outside the web root): `mkdir -p ~/exp_data/representation_stability`.
3. Over SSH, set the admin password (see [Admin page](#admin-page)).

Each release:
1. `npm run build`, then `tar -czf dist.tar.gz --exclude=./review.html -C dist .` (the participant site without the review page).
2. Upload `dist.tar.gz` to `~/apps/<app>/` (FileZilla), then over SSH: `cd ~/apps/<app> && tar -xzf dist.tar.gz && rm dist.tar.gz`. When only code changed, uploading `index.html`, `admin.html`, `app/` and the three PHP files (`save.php`, `admin_api.php`, `data_config.php`) is enough.
3. Open `https://<site>/save.php?check=1`: it must show `"dataDirWritable":true`.

Collecting data: open `https://<site>/admin.html` and use **Download CSV**, or download `~/exp_data/representation_stability/*.jsonl` (FileZilla) to a local folder and run `npm run data:csv -- <folder>`. Both produce identical CSV files.

---

## Admin page

`admin.html` is the researcher's view of the collected data. It is not linked from the participant page (checked by `verify:participant-build`).

- **Sessions**: every session with participant ID, status (complete / in progress / interrupted), training trials and when the criterion was met, drift progress, drift accuracy and mean response time. The list refreshes every 30 s. It can be filtered by text and status; development sessions are hidden unless *Include development sessions* is ticked. **Download CSV (shown sessions)** writes one combined file with the same columns as `npm run data:csv`.
- **Session view**: summary tiles, a chart of rolling accuracy over the last 20 trials with the start of drift marked, the **image order** (every face in the order it was shown, numbered within its phase, with its group, the answer and ✓/✗; filter by phase, group and correctness; click a face for its details), the PCA stimulus plot, the full trial table, and a CSV download for that session. An unfinished session counts as *in progress* while its file changed in the last 30 minutes, and as *interrupted* after that.
- **Planned image order**: the faces every session sees, in order, for the fixed seed (shown read-only; it can only be changed in the configuration file) and the saved experiment settings. Stimulus selection is deterministic, so this is exact. Training images always come in this order (answers only decide how many are shown), and drift images are the same however long training took.
- **Settings**: the experiment settings below.

### Experiment settings

The **Settings** tab changes four values without a new build:

| Setting | Default | Notes |
| --- | --- | --- |
| Center A initial angle | 0° | Training center of A and start of the drift. |
| Center B initial angle | 180° | Always opposite A (decision D3: B = −A). Changing either angle moves the other. |
| Movement per drift trial | 1° | Both centers rotate counterclockwise after every drift answer. Must be > 0 and ≤ 360. |
| Number of drift trials | 360 | Filled in automatically for one full rotation, 360 ÷ movement, whenever the movement changes (rounded when 360 is not a multiple, with a warning). It can still be edited. |

- **Scope.** Saved settings apply to every session that starts afterwards; a running session keeps the settings it started with. Each session records them in its metadata (`experimentSettings`: values, save time and source), and they are part of `effectiveConfigJson` in the CSV. The session page shows the settings a session used.
- **Storage.** On the server, the settings are saved in `~/exp_data/representation_stability_settings.json` (outside the web root) by `admin_api.php`, after a login and server-side validation. Participants read them with `save.php?settings=1`. Without the file, the defaults of [src/config/experimentConfig.js](src/config/experimentConfig.js) apply. Under `npm run dev`, the settings are saved in this browser's localStorage.
- **Failure.** If the settings cannot be loaded, the participant sees an error and the session does not start, so a session never runs silently with the wrong settings.
- Development mode applies its own overrides on top of the saved settings.

**Where the data comes from.** On the server, the page reads the session files through [public/admin_api.php](public/admin_api.php), which requires a login and never changes session files (its only write is the settings file). Under `npm run dev` (no PHP), it shows the sessions saved in this browser's localStorage, without a login.

**Setting the password** (once, over SSH on the server). Run this, then type the password and press Enter:

```bash
php -r 'echo password_hash(trim(fgets(STDIN)), PASSWORD_DEFAULT), PHP_EOL;' > ~/exp_data/representation_stability_admin_password
```

Only the hash is stored, outside the web root. Run the same command again to change the password. Without this file the login is disabled and the page says so.

**Security.** The login uses a PHP session cookie (`HttpOnly`, `SameSite=Strict`, `Secure` over HTTPS) that expires after 2 hours of inactivity or 12 hours in total. After 5 wrong passwords, the IP address is locked out for 15 minutes (`~/exp_data/representation_stability_admin_attempts.json`). Session IDs are validated before any file is read. The API never writes or deletes session data; saving settings requires a login.

---

## Development mode

Development mode is off by default and opens in one of two ways:
- `npm run dev` with `?dev=1`. Production builds ignore `?dev=1`.
- Setting `development.enabled: true` in the configuration.

Its code is loaded lazily and never appears in participant mode.

- **Setup screen**: overrides for minimum training trials, accuracy window, threshold, drift trial count, and degrees per trial. The fixed session seed is shown but cannot be changed. Overrides are validated and recorded in `developmentOverrides` and `effectiveConfig`.
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

Filters cover phase (all/training/drift) and group (all/A/B). Clicking a point shows its details and a thumbnail. **Export SVG** and **Export PNG** save `<sessionId>_pca_plot.svg|png`.

The review page reads sessions from this browser's `localStorage` and updates live while the experiment runs in another tab. It can also load an exported CSV.

---

## Verification results

Results as of 2026-10-28: `npm test` 114/114 passed, `npm run test:e2e` 2/2 passed (Chrome; the development test now also opens the admin page), `verify:repo` OK, `verify:participant-build` OK. `save.php` and `admin_api.php` were also exercised with PHP 8.3 in the Opalstack folder layout: a full 380-trial session, an interrupted session, login, wrong password, the lockout after 5 failures, sign-out, and the session CSV (byte-identical to `npm run data:csv`).

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
| 11. CSV | One row per response, every column, RFC 4180 escaping, metadata columns, `_INCOMPLETE` partial exports. By researcher decision (2026-10-28), participants get no download: the CSV comes from the admin page or `npm run data:csv`. |
| 12. Identity and consent | `PROLIFIC_PID` used when non-empty, else anonymous; `Continue` disabled until consent; bypass only in development mode and recorded honestly. |
| 13. No dev in participant mode; replaceable adapter | e2e on the production build with `?dev=1` shows no development controls; `verify:participant-build`; engine runs unchanged on the memory and localStorage adapters. |

---

## Known limitations

- **Reload restarts**: a participant who reloads the page starts a new session from consent; the interrupted session stays on the server as a separate `_INCOMPLETE` file with the same `participantId`.
- **Placeholder consent text**: the consent text is a placeholder and `completion.prolificCompletionUrl` is empty; both must be completed by the researcher.
- **Onset precision**: onset is tied to the browser's rendering frame (about 16 ms resolution at 60 Hz). Actual display latency of the monitor is not measurable from the browser.
- **Size**: `public/assets/faces/` (85.7 MB, 11,817 images) is regenerated by `prepare-assets` and is not tracked. `public/assets/pca.json` and the manifest are small derived files.

## Directory layout

```text
human_experiment/
├── index.html, review.html          participant page, researcher review page
├── admin.html                       admin page (sessions, image order, data)
├── src/
│   ├── main.js, app.js, review.js   composition roots and screen rendering
│   ├── admin.js, admin/             admin page: data sources, session list/view, chart, planned order
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
├── public/save.php                  server data endpoint (copied to dist/)
├── public/admin_api.php             read-only admin data endpoint (login required)
├── public/data_config.php           data folder and admin password file locations
├── public/assets/                   derived PCA JSON, manifest, copied faces
└── tests/                           unit, integration (jsdom), e2e (Playwright), parity fixtures
```
