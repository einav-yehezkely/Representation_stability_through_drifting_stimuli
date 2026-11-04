# Human Experiment 2 — Similarity Choices Under Drifting Stimuli

A browser experiment for human participants, built from [human_experiment](../human_experiment/README.md). In each trial a face appears and disappears, then two faces appear side by side, and the participant clicks the one more similar to the face just seen. The two faces come from opposite ends of the PCA space (groups A and B). One of them becomes the face to compare with in the next trial, while both group centers rotate along the trajectory of the first experiment.

The two sites share no code at run time. This folder is a separate copy with its own data folder, admin password, settings and `localStorage` keys, so the sites can run side by side.

> **Before running on Prolific**, replace the consent placeholders in [src/content/consent.js](src/content/consent.js), review the instructions in [src/content/instructions.js](src/content/instructions.js) and set `completion.prolificCompletionUrl`.

---

## Quick start

All commands run inside `human_experiment_2/`. Requirements: Node.js ≥ 20. Google Chrome is needed only for the end-to-end tests.

```bash
npm install                 # dev dependencies (vite, vitest, jsdom, @playwright/test)
npm run prepare-assets      # copy faces + derive PCA JSON into public/assets/ (read-only on the source);
                            # restart a running `npm run dev` afterwards so it serves the new faces
npm run dev                 # http://localhost:5174/            participant view
                            # http://localhost:5174/?dev=1      development mode
                            # http://localhost:5174/review.html researcher review page
                            # http://localhost:5174/admin.html  admin page (sessions saved in this browser)
npm run build               # production build -> dist/ (static files, ~105 MB incl. faces)
npm run preview             # serve dist/ on http://localhost:4174/
```

The ports differ from the first site (5173, 4173). To serve the study, deploy `dist/` to a web server with PHP (see [Deployment](#deployment-opalstack)). Participant links can carry `?PROLIFIC_PID=...`.

| Script | Purpose |
| --- | --- |
| `npm run prepare-assets` | Reads `../pca_top2_filtered_female_vgg_1.csv` and `../female_faces/` (read-only). Writes `public/assets/faces/`, `public/assets/pca.json` and `public/assets/assets-manifest.json`. Fails loudly if an image is missing. |
| `npm test` | Unit and integration tests (Vitest, jsdom). Requires `prepare-assets` first. |
| `npm run test:e2e` | Real-browser tests in the installed Chrome (Playwright). Run `npm run build` first. |
| `npm run parity:fixtures` | Regenerates `tests/fixtures/parity/parity.json` from verbatim copies of the source Python functions. |
| `npm run verify:participant-build` | Confirms that development mode is disabled and lazily loaded in `dist/`. |
| `npm run data:csv -- <folder>` | Converts the server's `*.jsonl` session files to CSV (one file per session plus `all_sessions.csv`) in `<folder>/csv/`. |

---

## Participant flow

`Consent → Instructions → Trials → Completion`

There is no training phase and no feedback.

- **Consent**: the study text in [src/content/consent.js](src/content/consent.js) consists of `[RESEARCHER: ...]` placeholders, which must be replaced with approved text. `Continue` stays disabled until the box is checked.
- **Instructions**: [src/content/instructions.js](src/content/instructions.js). They do not mention groups or the drift.
- **Trial** (360 by default):
  1. The **sample** face, alone in the center, for 1000 ms.
  2. An empty screen for 500 ms.
  3. **Two faces** side by side with the question "Which face is more similar to the previous face?". They wait for a click, with no time limit. The faces become clickable when they appear, and the first click counts.
  4. An empty screen for 500 ms, counted from the click. The answer is saved and the next images load during it. Both centers have moved by the next trial. After the last trial, the completion screen follows directly.
- **Completion**: "Your responses have been saved" and, when `completion.prolificCompletionUrl` is set, a `Return to Prolific` button. Participants are never offered a download of their data.
- **Recovery**: when the page is reloaded during a session, the interrupted session keeps its saved data, is marked `interrupted`, and a new session starts.

All participant-facing text is in [src/content/](src/content/).

---

## Trial sequence

The sequence is built in [src/experiment/sequencePlanner.js](src/experiment/sequencePlanner.js). It depends only on the seed and the configuration. **Answers never change it**, so every participant sees the same faces in the same order.

| Element | Rule |
| --- | --- |
| Two faces of trial *k* | One from each group: the nearest unused image to center A, then to center B, at the centers of trial *k* |
| Sample of trial 1 | The image of center A (nearest to the center, distance 0), group A |
| Sample of trial *k* > 1 | One of the two faces of trial *k − 1*: the one of the group named by the sample schedule |
| Sample schedule | Trials 2…N: one shuffled list with A and B balanced (\|A − B\| ≤ 1), fixed by the seed |
| Sides | Which group is on the left: one shuffled list with A and B balanced, fixed by the seed |
| Correct answer | The face of the sample's group, so `correct = chosenGroup === sampleGroup`. Recorded, never shown |
| Centers | Rotate counterclockwise by `degreesPerTrial` after each answer. Trial *k* uses `initialAngle + (k − 1) · degreesPerTrial` |
| Repeats | None: an image is selected once per session. The only reuse is the continuity itself, where a face returns as the next sample |

Because the centers move after the answer, a sample was selected at the previous position of its center, while the two faces are selected at the current one.

**Which group continues.** The sample switches group whenever the schedule says so, whatever the participant chose. The participant therefore does not follow group A alone. With the default seed, the sample changes group between 189 of the 359 consecutive pairs of trials.

### Measured consequences (default configuration and seed)

Computed from the real plan:

- 360 trials use 721 different images (2 per trial, plus the first sample), out of 11,817.
- Distance from a selected face to its center: median 0.008, 90th percentile 0.086, maximum 0.137.
- Distance from the sample to the face of its own group: median 0.011, maximum 0.211.
- Distance from the sample to the face of the other group: median 0.907, minimum 0.767.
- In all 359 trials with a sample from the previous trial, the correct face is the nearer one in the PCA plane.
- Sample A/B: 180/180. Group A on the left in 180 of 360 trials.
- Fixed display time is 2 s per trial, so about 12 minutes in total, plus the participant's response time.

---

## Scientific logic and provenance

The trajectory, centers and nearest-image selection are unchanged from the first site, which documents their source in the neural-network experiment. See [human_experiment/README.md](../human_experiment/README.md#scientific-logic-and-provenance).

- **Coordinates**: PC1 = x, PC2 = y, origin (0, 0). PCA file: `pca_top2_filtered_female_vgg_1.csv`.
- **Angles**: `degrees(atan2(PC2, PC1)) mod 360`.
- **Rotation**: counterclockwise rotation matrix, applied incrementally to the center vectors.
- **Center A**: the actual data point minimizing `angle_error + 100·|r − 0.45|` for the configured angle. For 0° this is `183785.jpg`.
- **Center B**: `−A`, always 180° from A.

[tests/unit/parity.test.js](tests/unit/parity.test.js) checks these functions against fixtures generated by the source Python functions. [reference_source/](reference_source/) and [scripts/reference/](scripts/reference/) are copies of the source code, kept for those fixtures and for documentation.

### Researcher decisions

| Decision | Implementation |
| --- | --- |
| Two faces on screen; the sample is shown before them and is not on screen with them | [src/components/trialScreen.js](src/components/trialScreen.js) |
| Sample shown for 1000 ms, then a 500 ms blank | `timing.sampleDurationMs`, `timing.blankDurationMs` |
| 500 ms blank between the click and the next sample | `timing.interTrialDurationMs` |
| Sample of each trial is a face of the previous trial, A or B at random | `sampleSchedule` |
| First sample is the image of center A | `FIRST_SAMPLE_GROUP`, [sequencePlanner.js](src/experiment/sequencePlanner.js) |
| No training phase, no feedback | Single trial loop in [experimentEngine.js](src/experiment/experimentEngine.js) |
| Answer by clicking a face | `response` is `left` or `right` |
| Sides random and balanced | `leftSchedule` |
| Centers move after each answer | [driftPhase.js](src/experiment/driftPhase.js) |
| No repeated images | `scientific.selection: "nearestUnused"`, one used set per session |

### Values carried over from the first site

The researcher did not set these separately. They keep the first site's values and can be changed: 360 trials, 1° per trial, start at 0°/180° (admin Settings), the fixed seed 20261014, and a display scale of ×2.

---

## Configuration

Single entry point: [src/config/experimentConfig.js](src/config/experimentConfig.js), validated before the experiment starts ([src/config/validateConfig.js](src/config/validateConfig.js)). The effective configuration, including development overrides, is stored in every session's metadata.

| Key | Default | Notes |
| --- | --- | --- |
| `drift.initialAngleA` / `initialAngleB` | 0 / 180 | Must be 180° apart. Defaults only: the admin page can change them. |
| `drift.numberOfTrials` / `degreesPerTrial` | 360 / 1 | Defaults only: the admin page can change them. At most 5,908 trials (two new images per trial, no repeats). |
| `timing.sampleDurationMs` / `blankDurationMs` / `interTrialDurationMs` | 1000 / 500 / 500 | Sample, blank before the two faces, blank after the click. |
| `scientific.*` | see file | Source-derived values plus `selection`. |
| `randomization.seedPolicy`, `randomization.fixedSeed` | `"fixed"`, 20261014 | One seed for all sessions (uint32). |
| `ui.imageDisplayScale` | 2 | Each face is up to 356×436 px, smaller on small windows. |
| `ui.preloadCount` | 3 | Trials whose images are downloaded in advance (performance only; 0 disables). |
| `storage.adapter` | `server` | `localCsv` (pilot) and `memory` (tests) are also available. |
| `storage.developmentAdapter` | `localCsv` | Used by the Vite dev server, which cannot run PHP. |
| `storage.serverEndpoint`, `requestTimeoutMs`, `maxAttempts` | `save.php`, 15000, 3 | Attempts per save before the participant sees `Try again`. |
| `storage.localStoragePrefix` | `hx2` | Differs from the first site (`hx`). |
| `completion.prolificCompletionUrl` | `""` | Prolific completion link. Empty: no link is shown. |
| `development.enabled` | `false` | See [Development mode](#development-mode). |

### Randomization and reproducibility

- **Generator**: sfc32 v1, with two independent named streams seeded from `(sessionSeed, name)`: `sampleSchedule` and `sideSchedule`.
- **Image selection**: deterministic (no randomness).
- **Stored for reproduction**: seed, algorithm and version, both schedules, the effective configuration, and the PCA SHA-256 and face-list hash.
- `Math.random()` is not used anywhere (checked by a test).

---

## Timing

Measured with `performance.now()` ([src/components/trialScreen.js](src/components/trialScreen.js)):

1. All three images of the trial are fetched and fully decoded off-screen before anything is shown, so loading never stretches the sample, the blank or the response time. Images of the next trials are preloaded while an answer is being saved.
2. Each step changes the page and waits for the second `requestAnimationFrame`, after that frame has been rendered. The time of the step is taken there.
3. When the two faces are painted, `onset = performance.now()` is taken and the faces become clickable at the same moment.
4. On click, `performance.now()` is captured first. `responseTimeMs = click − onset`.
5. The screen is cleared at the click and the 500 ms timer starts. The answer is saved and the next images load during this time, so they lengthen the pause only if they take longer than it.

The three durations are timers plus the rendering frames, so the real values are slightly longer. Each trial records the measured values in `sampleDurationMs`, `blankDurationMs` and `interTrialDurationMs`.

A failed image load shows an English error with `Try again`. It creates no record, shows nothing of that trial and does not move the centers. Duplicate clicks are ignored, and the record key `sessionId:trialNumber` is idempotent.

---

## Data

### Trial record and CSV columns

`<sessionId>.csv` has one row per answered trial and a fixed header. Booleans are written as `true`/`false`, angles in degrees [0, 360), times in milliseconds, and timestamps in ISO 8601 UTC.

| Column | Meaning |
| --- | --- |
| `participantId`, `sessionId` | `PROLIFIC_PID` URL parameter if present, else `anon-<16 hex>`; session UUID |
| `trialNumber` | One-based index |
| `sampleGroup` | Group of the sample |
| `sampleSourceTrial` | Trial whose face became this sample; empty for trial 1 |
| `sampleImageId`, `sampleImageIndex`, `samplePc1`, `samplePc2`, `sampleAngle`, `sampleDistanceToCenter` | The sample. The distance is to its center at the time it was selected, so at the previous trial's position |
| `optionAImageId`, … `optionADistanceToCenter` | The face from group A, with the same six fields |
| `optionBImageId`, … `optionBDistanceToCenter` | The face from group B, with the same six fields |
| `leftGroup` | Group shown on the left |
| `currentAngleA`, `currentAngleB` | Nominal center angles of this trial |
| `response` | `left` or `right` |
| `chosenGroup`, `chosenImageId` | Group and image of the clicked face |
| `correct` | `chosenGroup === sampleGroup` |
| `responseTimeMs` | Two faces painted → accepted click |
| `sampleDurationMs`, `blankDurationMs` | Measured display times |
| `interTrialDurationMs` | Measured time from the previous click to this sample; empty for trial 1 |
| `timestamp` | Wall-clock time of the accepted response |
| `selectionRule` | `nearestUnused` |
| `centerA_pc1`, `centerA_pc2`, `centerA_actualAngle`, `centerB_*` | Exact center vectors and actual angles of this trial |
| `imageLoadFailures` | Failed load attempts before this trial was shown |

**Session metadata** (repeated on every row): `experimentVersion`, `schemaVersion`, `participantIdSource`, `sessionStartTime`, `sessionEndTime`, `consentGiven`, `consentTimestamp`, `consentBypassed`, `developmentMode`, `sessionSeed`, `rngAlgorithm`, `rngAlgorithmVersion`, `pcaSource`, `pcaSha256`, `facesListSha256`, `completedTrials`, `completionStatus` (`complete`, `in_progress`, `interrupted` or `incomplete_dev_early_finish`), `terminationReason`, `exportedAsIncomplete`, and the JSON columns `initialCentersJson`, `effectiveConfigJson`, `developmentOverridesJson` and `randomizationJson`.

### Storage

Trial logic depends only on the contract in [src/services/dataStorage/storageContract.js](src/services/dataStorage/storageContract.js): `startSession`, `saveTrial`, `completeSession`. Failures reject with `StorageError`; the engine shows them in English and does not advance, and `Try again` re-saves the same record.

- **Server adapter** ([serverStorage.js](src/services/dataStorage/serverStorage.js)): POSTs each call to [public/save.php](public/save.php) and resolves only after the server confirmed the write. It also keeps a best-effort copy in `localStorage`. `save.php` appends to `~/exp_data/representation_stability_2/<sessionId>.jsonl`, one JSON object per line. A session without a `complete` record is exported as `interrupted` and `_INCOMPLETE`.
- **Pilot adapter** ([localCsvStorage.js](src/services/dataStorage/localCsvStorage.js)): keeps sessions in `localStorage` under `hx2:*`. Used by `npm run dev`.

---

## Deployment (Opalstack)

Use a separate application from the first experiment. One-time setup:
1. Opalstack dashboard: create an Application of type **PHP-FPM + Apache** and a Site that routes a path of the domain to it, with Let's Encrypt enabled.
2. Over SSH: `mkdir -p ~/exp_data/representation_stability_2`. The folder sits next to `apps/`, outside the web root.
3. Over SSH, set the admin password (see [Admin page](#admin-page)).

Each release:
1. `npm run build`, then `tar -czf dist.tar.gz --exclude=./review.html -C dist .`
2. Upload `dist.tar.gz` to `~/apps/<app>/`, then over SSH: `cd ~/apps/<app> && tar -xzf dist.tar.gz && rm dist.tar.gz`.
3. Open `https://<site>/save.php?check=1`: it must show `"dataDirWritable":true`.

Collecting data: open `https://<site>/admin.html` and use **Download CSV**, or download `~/exp_data/representation_stability_2/*.jsonl` to a local folder and run `npm run data:csv -- <folder>`.

---

## Admin page

`admin.html` is the researcher's view of the collected data. It is not linked from the participant page (checked by `verify:participant-build`).

- **Sessions**: every session with participant ID, status, trials answered, accuracy and mean response time. **Download CSV (shown sessions)** writes one combined file.
- **Session view**: summary tiles (accuracy overall and by sample group), rolling accuracy over the last 20 trials, the **image order** (each trial with its sample, left and right faces, the chosen face ringed and ✓/✗; click a trial for its details), the PCA plot of the samples colored by the group the participant chose, the trial table and a CSV download.
- **Planned image order**: all trials every session sees, for the fixed seed and the saved settings. This is exact, because answers do not change the sequence.
- **Settings**: the experiment settings below.

### Experiment settings

| Setting | Default | Notes |
| --- | --- | --- |
| Center A initial angle | 0° | Also sets the first sample. |
| Center B initial angle | 180° | Always opposite A. Changing either angle moves the other. |
| Movement per trial | 1° | Must be > 0 and ≤ 360. |
| Number of trials | 360 | Filled in automatically for one full rotation whenever the movement changes. It can still be edited. |

Saved settings apply to every session that starts afterwards, and each session records them. On the server they are saved in `~/exp_data/representation_stability_2_settings.json`. If they cannot be loaded, the participant sees an error and the session does not start. The three screen durations are not admin settings; they are in the configuration file.

**Setting the password** (once, over SSH on the server). Run this, then type the password and press Enter:

```bash
php -r 'echo password_hash(trim(fgets(STDIN)), PASSWORD_DEFAULT), PHP_EOL;' > ~/exp_data/representation_stability_2_admin_password
```

Only the hash is stored, outside the web root. Without this file the login is disabled and the page says so.

**Security.** As in the first site: an `HttpOnly`, `SameSite=Strict` session cookie (`hx2_admin`), expiry after 2 hours of inactivity or 12 hours in total, and a 15-minute lockout per IP address after 5 wrong passwords. The API never writes or deletes session data.

---

## Development mode

Off by default. It opens with `npm run dev` and `?dev=1`, or by setting `development.enabled: true`. Production builds ignore `?dev=1`. Its code is loaded lazily and never appears in participant mode.

- **Setup screen**: overrides for the number of trials, degrees per trial, and the three screen durations, recorded in `developmentOverrides` and `effectiveConfig`.
- **Consent bypass**: recorded as `consentBypassed: true, consentGiven: false`.
- **Debug panel**: step, trial, sample and its source trial, both options, left group, nominal and actual center angles, seed.
- **Finish early**, **Download partial CSV** (marked `_INCOMPLETE`), and the live **PCA plot**.

## Researcher visualization

Available in the development panel, on the admin session view and at `review.html`, which must not be shown to participants. It uses the plot conventions of the first site. Each point is the **sample** of a trial, colored by its group (development panel, review page) or by the group the participant chose (admin session view).

---

## Verification results

Results as of 2026-11-04:
- `npm test`: 121/121 passed.
- `npm run test:e2e`: 2/2 passed in Chrome. On the production build, the measured sample duration stayed within 995–1120 ms, and both blanks within 495–620 ms.
- `verify:participant-build`: OK.
- `save.php` and `admin_api.php` were run with PHP 8.3 in a container, in the Opalstack folder layout: 20/20 checks passed (saving a session, repeated trials, login, wrong password, session summary, settings and their validation). They have not been run on the real server yet, so do the deployment check and a test session there before collecting data.

## Known limitations

- **Reload restarts**: a participant who reloads the page starts a new session from consent. The interrupted session stays on the server as a separate `_INCOMPLETE` file with the same `participantId`.
- **Placeholder consent text** and an empty Prolific link must be completed by the researcher.
- **Onset precision**: onset is tied to the browser's rendering frame (about 16 ms at 60 Hz).
- **Size**: `public/assets/faces/` (85.7 MB, 11,817 images) is regenerated by `prepare-assets` and is not tracked.

## Directory layout

```text
human_experiment_2/
├── index.html, review.html, admin.html
├── src/
│   ├── main.js, app.js, review.js   composition roots and screen rendering
│   ├── admin.js, admin/             admin page
│   ├── config/                      experimentConfig.js (single entry point), validateConfig.js
│   ├── content/                     researcher-editable English text
│   ├── components/                  screens, trial screen / presenter
│   ├── experiment/                  engine, sequence planner, drift, response coding, randomization
│   ├── scientific/                  PCA loader, trajectory, cluster construction, stimulus selector
│   ├── services/dataStorage/        contract, adapters, CSV serializer
│   ├── visualization/               PCA plot, viewer, export, trajectory membership
│   └── dev/                         development setup screen and panel (lazy)
├── scripts/                         prepare-assets, verify script, reference/ (Python parity)
├── reference_source/                verbatim copies of the inspected source files
├── public/                          save.php, admin_api.php, data_config.php, assets/
└── tests/                           unit, integration (jsdom), e2e (Playwright), parity fixtures
```
