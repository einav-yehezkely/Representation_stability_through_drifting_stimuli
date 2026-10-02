# Human Experiment PRD — Representation Stability Under Drifting Stimuli

## 1. Objective and implementation scope

Build an English-language browser experiment for human participants, parallel to the existing neural-network experiment. Participants learn to classify faces into Group A or Group B with supervised feedback, then continue classifying without feedback while the category centers move along the existing scientific PCA trajectory.

This document is an implementation specification for Claude Code. Implement a working, modular experiment, local pilot persistence, CSV export, development controls, and verification of the requirements below. Do not invent scientific procedures that are absent from the existing experiment.

## 2. Absolute repository protection rule

**ALL EXISTING REPOSITORY CODE AND FILES ARE READ-ONLY. THERE ARE NO EXCEPTIONS.**

Claude must NEVER modify, delete, move, rename, refactor, reformat, or overwrite any existing file. Do not change existing configuration, dependencies, lockfiles, paths, data, assets, or repository structure. Do not move the neural-network experiment into a new directory. Do not run tools that rewrite existing files.

**All new implementation work must be created under `human_experiment/`.** This includes application code, configuration, package manifests and lockfiles, scripts, generated data, build output, documentation, tests, and copied assets. Run installation and build tools from that directory, with output confined to it.

If existing code or logic is needed, first copy it into `human_experiment/`, then modify only the copy. Existing PCA data and `female_faces/` may be read directly; if browser serving requires copies or conversion, create those copies or derived files only inside `human_experiment/`. Preserve source files and their original paths.

Before and after implementation, inspect repository status and verify that no pre-existing file has changed. Preserve any changes that already existed before work began. This delivered PRD is a specification artifact; it does not authorize subsequent edits outside `human_experiment/`.

## 3. Mandatory scientific reference inspection

Before implementing stimulus generation, inspect the existing neural-network experiment and locate:

- The existing `female_faces/` directory and image identifiers.
- The existing PCA data, loading procedure, dimensions, preprocessing, and image-to-coordinate mapping.
- The scientific trajectory: origin, geometry, radius, coordinate conventions, and angular conventions.
- How Group A and Group B clusters are constructed around their current centers.
- Candidate membership, distances, tolerances, sampling weights, and any repeat or exclusion rules.
- Rotation direction, transformation, and center-update procedure.
- The existing face/stimulus-selection procedure.

Reproduce that experiment's PCA trajectory, cluster construction, rotation, and stimulus-selection logic. Adapt presentation to **one face per human trial**, without substituting a new scientific algorithm. Do not independently assume nearest-neighbor selection, a radius, an origin at `(0, 0)`, or an `atan2(PC2, PC1)` angle convention unless supported by the source experiment.

Document the inspected source paths and corresponding copied/adapted modules in `human_experiment/README.md`. Put source-derived scientific parameters in central configuration. If required code or data is missing or genuinely ambiguous, report the specific dependency rather than silently substituting an invented procedure. No new scientific decisions are required by this PRD beyond configurable placeholders and source-derived parameters.

## 4. Participant flow and English UI

The flow is:

`Consent → Instructions → Training with feedback → Transition → Drift without feedback → Completion and CSV download`

### Consent

Show editable English study-information content and the checkbox:

> I have read the information above and agree to participate in this study.

The `Continue` button must remain disabled until consent is checked. Record consent and its timestamp in session metadata. Keep the study-information text in a separate content file with clearly marked researcher-editable placeholders; do not invent approval numbers, institutional claims, or finalized consent language. Development mode may explicitly bypass consent; record the bypass rather than recording consent as given.

### Instructions

Explain that participants will see one face at a time and learn to classify faces into A or B. Explain all four response options and that initial trials provide feedback. Keep wording editable independently of scientific logic. Do not disclose the subsequent category drift in participant-facing instructions.

### Trial screen

Display exactly **one face image**, centered, with four response buttons below it in this fixed order:

`A / Probably A / Probably B / B`

Keep the layout clean and minimally distracting. Do not shuffle button order. A trial presents a face selected from one true group, A or B; never show a face pair or one image from each group simultaneously. Disable response controls until the image is displayed and immediately after the first accepted response. Accept and save only one response per trial.

### Transition screen

After the qualifying training trial and its feedback, show:

> **The first part of the experiment is complete.**
>
> In the next part, you will continue performing the same task, but you will no longer receive feedback about your answers.
>
> Please continue to classify each face as accurately as possible.
>
> **Continue**

Do not begin drift until the participant clicks `Continue`. The screen must explicitly announce that feedback stops; it must not announce category movement.

## 5. Response coding and correctness

Preserve the original four-level response and derive a separate binary response:

| Original response | Binary response |
| --- | --- |
| A | A |
| Probably A | A |
| Probably B | B |
| B | B |

`correct = (binaryResponse === trueGroup)`. Confidence does not affect binary correctness. `trueGroup` is the group used to generate/select that trial's stimulus, according to the source cluster logic; do not relabel stimuli from participant responses.

Compute and save correctness in both phases. Show correctness feedback only during training. Drift must not reveal the true group, correctness, or other outcome cues through labels, colors, or debug information in participant mode.

## 6. Training phase

- Default center angles are independently configurable: A = `0°`, B = `180°`.
- Both centers remain completely fixed for the entire training phase. There is no training drift.
- Do not hard-code B as A + 180°. The default separation is 180°, but both initial angles are independent parameters.
- Select one group and one source-compatible face per trial using seeded, randomized, balanced group selection.
- After each accepted response, show English feedback such as `Correct` or `Incorrect`, then advance. Keep feedback timing/advancement behavior centrally configurable.
- After each completed training response, evaluate the learning criterion.

**Default training ends only when BOTH conditions hold:**

1. At least `20` training trials have been completed.
2. Binary accuracy over the **LAST 20 completed training trials** is `>= 0.80`.

This is a sliding window, not cumulative accuracy, and includes the just-completed trial. At the default window size, at least 16 of its 20 responses must be correct. Do not evaluate an incomplete window. If the criterion fails, continue training with feedback and fixed centers, reevaluating after every response.

```text
eligible = completedTrainingTrials >= minTrials
           AND completedTrainingTrials >= accuracyWindow
accuracy = correctCount(last accuracyWindow training trials) / accuracyWindow
trainingPassed = eligible AND accuracy >= accuracyThreshold
```

Do not introduce a production trial cap, timeout, or alternate route that treats a participant as trained without meeting the criterion. Early finishing is a development control and must be marked as incomplete. The qualifying training trial still receives feedback before transition.

## 7. Drift phase and exact movement order

Defaults:

- `numberOfTrials = 360`
- `degreesPerTrial = 1`
- Initial centers are the same configured centers used during training: A = 0°, B = 180° by default.

Both centers move together using the source experiment's rotation convention. Their configured angular separation is preserved. The positive movement convention must match the inspected source; do not independently choose a clockwise/counterclockwise scientific convention.

**The first drift trial uses the initial positions. Move centers only AFTER each response, never before the first display.** For one-based drift trial index `k`, the angular progression is `(k - 1) × degreesPerTrial`, applied through the source rotation convention, with recorded angles normalized consistently to `[0, 360)`.

Under the default positive angular progression:

| Drift trial | Displayed A angle | Displayed B angle |
| --- | --- | --- |
| 1 | 0° | 180° |
| 2 | 1° | 181° |
| 180 | 179° | 359° |
| 181 | 180° | 0° |
| 360 | 359° | 179° |

After response 360, the internal next centers return to 0°/180°; there is no additional displayed trial. Thus default displayed A positions are 0° through 359°, and the final post-response update completes the full rotation. Store the centers that generated the displayed face, not the next centers after movement.

Each drift trial must:

1. Snapshot current A/B centers.
2. Select A or B using seeded, balanced randomization.
3. Select exactly one face with the copied scientific stimulus-selection logic.
4. Load and display it, then begin response timing.
5. Accept one response and create/save its record using the display-time snapshot.
6. Advance both centers by the configured movement amount after that response.
7. Show the next trial, or complete after the configured number of drift responses.

One movement/trial corresponds to one displayed face from one selected group. Do not double trial count by showing both groups at each angle. No correctness feedback is shown during drift. Trial count and movement amount are independently configurable; do not force their product to 360°.

## 8. Randomization and reproducibility

Create and store a session seed. Use one explicitly seeded randomization system for group scheduling, selection among eligible faces, and all other experimental random decisions. Do not use untracked `Math.random()` for experiment decisions.

Use seeded shuffled balanced schedules: for the fixed drift phase, A/B counts differ by at most one; for open-ended training, generate balanced shuffled blocks with centrally configured block size. Preserve source face-selection sampling and repeat rules. Save the effective schedule or sufficient generator state, seed, algorithm/version, configuration, and asset/PCA version information to reproduce the session. Document the balancing policy. Reproduction of adaptive training also requires the recorded participant responses.

## 9. Response timing and loading

Measure response time in milliseconds with the monotonic `performance.now()` clock. Timing starts only when the current image has loaded/decoded and is actually presented, using a documented render/paint-aware onset procedure. Do not start at trial creation, fetch initiation, or before replacing the previous face. Enable responses at that same onset.

Capture response time at the first accepted participant action, before persistence or feedback work. Image-loading delays, feedback duration, and storage latency must not enter response time. Preloading is allowed without changing trial selection or onset semantics. Failed image loads must not count as answered trials, advance centers, or start a response timer; show an English recoverable error.

## 10. Central configuration

Keep all experimental parameters in one modular configuration entry point. No scientific magic numbers in UI, phase, or storage code. An illustrative shape is:

```javascript
const experimentConfig = {
  training: {
    initialAngleA: 0,
    initialAngleB: 180,
    minTrials: 20,
    accuracyWindow: 20,
    accuracyThreshold: 0.80,
  },
  drift: { numberOfTrials: 360, degreesPerTrial: 1 },
  assets: {
    facesDirectory: "<existing female_faces path or isolated served copy>",
    pcaDataPath: "<existing PCA path or isolated derived copy>",
  },
  scientific: { /* source-derived trajectory, cluster and selection parameters */ },
  randomization: { /* seed policy, algorithm, balanced training block size */ },
  ui: { /* feedback timing, image presentation settings, editable content */ },
  storage: { adapter: "localCsv" },
  development: { enabled: false /* explicit development-only overrides */ },
};
```

Resolve source-derived placeholders by inspecting the repository. Validate finite angles, positive integer trial/window counts, a threshold in `[0, 1]`, valid data paths, and supported source parameters before starting. Record the resolved effective configuration, including development overrides, in session metadata. Drift starts from the configured training centers; avoid duplicate conflicting initial-position settings.

## 11. Participant and session identity

Create a unique `sessionId` for each new session. Read a nonempty `PROLIFIC_PID` URL parameter as `participantId` when present; otherwise generate an anonymous development identifier. Record the identifier source. Do not collect names, email addresses, or unnecessary identifying data, and do not store the full URL when only the participant parameter is needed.

Session metadata must include at least:

- Session and participant IDs, identifier source, session start and end timestamps.
- Consent status and timestamp, or explicit development bypass status.
- Session seed and reproducibility information.
- Effective configuration, experiment/schema version, and source-data version references.
- Development-mode flag, completed training and drift counts, and training criterion outcome.
- Completion status and termination reason; development early finish is incomplete.

Use consistent ISO 8601 UTC wall-clock timestamps for metadata and trial records. Wall-clock timestamps are separate from monotonic response-time measurement.

## 12. Trial data contract

Save one immutable record per answered trial with at least:

| Field | Meaning |
| --- | --- |
| `participantId`, `sessionId` | Session identity |
| `trialNumber` | One-based overall answered-trial index |
| `phaseTrialNumber` | One-based index within training or drift |
| `phase` | `training` or `drift` |
| `imageId` | Stable source image identifier |
| `trueGroup` | Group A or B selected for this stimulus |
| `pc1`, `pc2` | Actual image coordinates in the source PCA representation |
| `stimulusAngle` | Actual stimulus angle using the source trajectory convention |
| `currentAngleA`, `currentAngleB` | Center angles used for this display |
| `response` | Original four-level response, preserved verbatim |
| `binaryResponse` | Derived A or B |
| `correct` | Binary correctness, also stored during drift |
| `responseTimeMs` | Time from displayed image onset to accepted response |
| `feedbackShown` | Whether correctness feedback was presented |
| `timestamp` | Accepted response wall-clock timestamp |

Retain any additional source coordinates needed to reconstruct selection and document the schema. Define angles and units clearly. The selected group center angle is not a substitute for the actual stimulus angle. Capture the display snapshot before any rotation. Use a stable record key such as session ID plus overall trial number to prevent duplicate records on retries.

## 13. Replaceable storage architecture

Scientific and UI logic must depend only on a storage interface:

```javascript
await storage.startSession(sessionMetadata);
await storage.saveTrial(trialData);
await storage.completeSession(completionMetadata);
```

Define these contracts, including error behavior, centrally. Keep browser storage, CSV serialization, downloads, and future network clients inside adapters. Select the adapter in configuration/composition code. A future server API or Supabase adapter must implement the same interface without changing phase logic, stimulus selection, timing, or response coding. Do not implement a backend or require Supabase credentials in this initial version.

### Initial development/pilot adapter

- `startSession()` initializes in-memory state and a session-scoped `localStorage` snapshot.
- `saveTrial()` saves each completed trial to memory AND `localStorage`, updating relevant session state. Do not download a CSV after every response.
- `completeSession()` saves final metadata and generates/downloads the CSV from the retained trial records.
- Handle storage errors visibly in English; do not silently claim successful persistence.
- Keep session records separate and do not overwrite or discard earlier sessions automatically.
- On refresh, preserve already saved records and provide recovery/export of the interrupted session. Do not silently start over and destroy it. Automatic experiment resumption is not required; if implemented, restore phase, trial indices, angles, seed/generator state, and response history faithfully.

**CSV/localStorage is only the initial development and pilot persistence mechanism. It is NOT the final Prolific data collection mechanism.** A browser download stays on the participant's computer and does not automatically deliver results to the researcher. Final online collection requires a later server-backed adapter and collection workflow.

## 14. CSV and completion

At normal experiment completion, automatically download `experiment_<sessionId>.csv`, with exactly one row per answered trial and stable headers containing the full trial schema. Include session metadata as repeated explicit columns or provide a documented companion metadata export; seed, configuration, consent, and completion metadata must remain exportable rather than existing only in memory. Serialize nested configuration deterministically where needed.

Escape commas, quotes, and newlines correctly, preserve the four original response strings, and use unambiguous booleans and numeric units. Do not use a participant name in the filename. Show an English completion screen with a manual download/retry button if automatic download is blocked. Development exports of partial sessions must be marked incomplete. Keep saved local data available after export.

## 15. Development mode

Provide an explicit development mode, off by default, with controls to:

- Bypass consent with an accurately recorded bypass flag.
- Shorten training by overriding minimum trials AND the accuracy window.
- Change accuracy threshold, drift trial count, and movement per trial.
- Display group, angles, phase, trial number, and useful debug information.
- Finish early, saving an incomplete session with a development termination reason.
- Download a partial CSV at any point after data exists.

All effective overrides must be recorded. Production/participant mode must hide debug information and development controls and use the default criterion unless centrally reconfigured. Development controls must not accidentally count an early finish as successful criterion completion.

## 16. Suggested isolated project structure

Use a simple appropriate web stack with its own tooling under `human_experiment/`; do not alter repository-root tooling. Exact file extensions may follow the chosen stack.

```text
human_experiment/
├── README.md
├── package.json
├── public/
│   └── assets/                 # served copies only if needed
├── scripts/                   # read-only source conversion/copy tools
├── src/
│   ├── config/
│   │   └── experimentConfig
│   ├── content/
│   │   ├── consent
│   │   └── instructions
│   ├── components/            # consent, instructions, trial, transition, end
│   ├── experiment/
│   │   ├── experimentEngine
│   │   ├── trainingPhase
│   │   ├── driftPhase
│   │   ├── responseCoding
│   │   └── seededRandomization
│   ├── scientific/
│   │   ├── pcaLoader
│   │   ├── trajectory
│   │   ├── clusterConstruction
│   │   └── stimulusSelector
│   └── services/
│       └── dataStorage/
│           ├── storageContract
│           ├── index
│           ├── localCsvStorage
│           └── csvSerializer
└── tests/
```

Scientific modules must be independent of UI and persistence. UI renders engine state and submits responses. The engine manages phase progression and calls the storage contract. Adapter-specific code must not decide scientific progression. Keep implementation practical and avoid unnecessary infrastructure.

## 17. Required acceptance checks and handoff

Verify and document these behaviors:

1. No pre-existing repository file was modified; all implementation output is under `human_experiment/`.
2. Scientific trajectory, cluster construction, rotation, and stimulus selection match the inspected source on representative inputs; document source-to-copy correspondence.
3. Exactly one face appears per trial, with the four fixed-order response buttons and preserved original responses.
4. Training centers stay fixed. Training cannot pass before 20 responses by default; 16/20 passes, 15/20 fails, and a session with high cumulative accuracy but a failing last-20 window does not pass.
5. The passing training response receives feedback, followed by the explicit no-feedback transition; drift waits for `Continue`.
6. Drift starts at the initial centers, moves after each response, displays the default 0°–359° progression, and ends after exactly 360 answered drift trials. Independent center configuration and alternate trial counts/step sizes work.
7. Drift records correctness but never displays correctness feedback.
8. A delayed image load does not inflate response time; a failed load does not create an answered trial or move centers. Duplicate clicks do not create duplicate responses.
9. The same seed, source data, configuration, and adaptive response history reproduce group scheduling and stimulus choices; balancing behaves as documented.
10. Trial records and session metadata are preserved in memory/localStorage and recoverable for export after refresh. Storage failures are surfaced.
11. Completion exports a valid CSV with one row per response, all required fields, and exportable metadata; manual retry and partial development export work.
12. `PROLIFIC_PID` is used when present; otherwise an anonymous ID is generated. Consent is enforced in participant mode, and development bypasses are recorded honestly.
13. Development controls and debug information are absent in participant mode. Replacing the storage adapter requires no edits to scientific or phase logic.

Deliver run/build instructions, configuration documentation, a trial/session schema description, source logic provenance, verification results, and the explicit pilot-only storage limitation in `human_experiment/README.md`. Do not claim readiness for final Prolific collection while using only local downloads.
