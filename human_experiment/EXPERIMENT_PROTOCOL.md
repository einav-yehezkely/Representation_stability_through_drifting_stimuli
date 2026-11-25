# Experiment Protocol — Representation Stability Under Drifting Stimuli (Human Experiment)

This protocol describes the experiment as currently implemented in `human_experiment/` (experiment version 1.0.0, data schema 1, as of 2026-11-11). The source of truth is the code. Values marked *default* can be changed by the researcher on the admin page without a new build. Every session records the values it actually used (`effectiveConfigJson`).

---

## 1. Aim

Humans learn to sort faces into two categories, A and B, with feedback. Then the two categories move slowly through face space without feedback. The question is whether participants' category representations follow the drifting stimuli or stay anchored to what they learned. The design follows the neural-network experiment in this repository (`network_classification/classify_rotation_resnet50.py`): same face space, same category centers, same rotation trajectory.

## 2. Participants and setting

- **Recruitment**: online, via Prolific. The participant link may carry `?PROLIFIC_PID=<id>`.
- **Device**: a desktop or laptop web browser. Answers are given by mouse click on the four buttons. Since 2026-11-24 the A and B keys can also be used for the answers "A" and "B" (the "Probably" answers have no key), and the instructions say so. Each trial records which input was used (`responseInput`: `click` or `key`).
- **Identity**: the participant ID is `PROLIFIC_PID` when present, otherwise a generated anonymous ID (`anon-<16 hex>`). No other personal information is collected. The page URL itself is not stored.
- **Sample size, inclusion and exclusion criteria, and compensation**: *not specified in the implementation; to be set by the researcher.*

## 3. Stimuli

- **Image set**: 11,817 face images (JPEG, 178 × 218 px) from `female_faces/`.
- **Face space**: each image has 2-D coordinates in `pca_top2_filtered_female_vgg_1.csv` (filename, PC1, PC2). These are the top two principal components of face embeddings of 118,165 `female_faces` images, keeping the 10% of images closest to the PC1–PC2 plane (smallest squared radius in components 3–39). File SHA-256: `cc393110…df6d6`.
- **Display**: one face at a time, centered, shown at 2× scale (356 × 436 px). On smaller screens it shrinks to at most 88% of the window width or 56% of its height, keeping the aspect ratio.
- **Angle convention**: angle = `atan2(PC2, PC1)` in degrees, in [0, 360); the origin is (0, 0).

## 4. Category structure

- **Center A**: the actual image that best matches the target angle (default 0°) at radius 0.45. It minimizes `angle error + 100 · |r − 0.45|`. With the defaults, this is `183785.jpg` at 359.49°, r = 0.4445.
- **Center B**: the point opposite A through the origin (`B = −A`). It is always 180° from A (default 180°).
- **Stimulus selection** (deterministic, one face per trial): the face shown on a trial is the image nearest (Euclidean distance in PC1–PC2) to the current center of that trial's group, excluding images already shown in the current phase.
  - Training and drift keep separate lists of used images. A face can appear once in training and once more in drift, but never twice within a phase.
  - An image counts as used only after its answer is saved.
- **Ground truth**: a trial's "true group" is the group whose center was used to select its face.

## 5. Procedure

`Consent → Instructions → Training (with feedback) → Transition → Drift (no feedback) → Completion`

All participant-facing text is in English.

### 5.1 Consent

A study information page, followed by the checkbox *"I have read the information above and agree to participate in this study."* `Continue` stays disabled until the box is checked. The consent time is recorded.

> ⚠️ The study information text (title, purpose, duration, risks, compensation, data handling, voluntariness, ethics approval and contact) is still a `[RESEARCHER: …]` placeholder in [src/content/consent.js](src/content/consent.js). It must be replaced with approved text before data collection.

### 5.2 Instructions

Participants are told:
- They will see faces one at a time.
- Each face belongs to Group A or Group B, and their task is to learn which.
- How to use the four answers (§6).
- They will get correct/incorrect feedback at first, and guessing at the start is expected.

The instructions **do not** mention that the categories will change.

### 5.3 Phase 1 — Training (with feedback)

- **Centers**: fixed at their initial positions.
- **Group order**: shuffled balanced blocks of 4 trials (2 A + 2 B).
- **Feedback**: after each answer, *"Correct"* or *"Incorrect"* is shown for 1,000 ms.
- **Learning criterion**: training ends as soon as both conditions hold:
  - at least 20 training trials have been completed, **and**
  - at least 80% of the **last 20** trials were correct (a sliding window, not cumulative accuracy).
- The trial that meets the criterion also gets feedback.
- **Limits**: there is no maximum number of training trials and no time limit.

### 5.4 Transition

The screen shows:

> *"The first part of the experiment is complete. In the next part, you will continue performing the same task, but you will no longer receive feedback about your answers. Please continue to classify each face as accurately as possible."*

Drift starts only after the participant clicks `Continue`. Category movement is never announced.

### 5.5 Phase 2 — Drift (no feedback)

- **Number of trials**: 360 (*default*).
- **Group order**: a single shuffled list with equal numbers of A and B trials (180 + 180 by default).
- **No feedback**: no feedback and no cues of any kind.
- **Center movement**:
  - The first drift trial uses the training centers.
  - **After** each answer is saved, both centers rotate together, counterclockwise around the origin, by 1° (*default*).
  - A and B therefore stay 180° apart. After 360 trials at 1° they have completed one full rotation.
- **Order within a trial**: select the group → select the face at the current centers → show it → accept the answer → save it → rotate the centers.

### 5.6 Completion

The final screen says *"Thank you! You have completed the experiment. Your responses have been saved."* When a Prolific completion URL is configured, it also shows a `Return to Prolific` button.

> ⚠️ `completion.prolificCompletionUrl` is currently empty, so no button is shown.

Participants are never offered a download of their data.

## 6. Trial structure and response

1. Response buttons are disabled and the previous face is removed.
2. The next face is loaded and decoded off-screen. Upcoming faces are preloaded, so this is usually instant.
3. The face appears. Once it has been painted on screen, the onset time is taken and the buttons are enabled at that same moment.
4. The participant clicks one of four buttons, always in this order: **`A` · `Probably A` · `Probably B` · `B`**. Instead of clicking `A` or `B`, the participant can press the A or B key. There is no response deadline.
5. Only the first answer counts, by click or by key. Response time = answer − onset, measured with `performance.now()`. It never includes image loading, saving, or feedback time.
6. The answer is saved to the server, and the next trial starts only after the server confirms the save. In training, the feedback is shown before the next trial.

**Response coding**:
- The four-level answer is stored as given.
- The binary answer counts `A` and `Probably A` as A, and `Probably B` and `B` as B.
- `correct` = binary answer equals the true group. Confidence does not affect correctness.
- `correct` is also computed for drift trials, even though no feedback is shown there.

There is no fixation cross and no fixed inter-trial interval. The gap between trials is the save round trip, plus 1 s of feedback in training.

## 7. Randomization and reproducibility

- **Seed**: all sessions use the same fixed seed, `20261014`, so every participant sees the same group order and the same faces in the same order. The seed can be changed only in [src/config/experimentConfig.js](src/config/experimentConfig.js).
- **Generator**: sfc32, with two independent streams: one for the training schedule and one for the drift schedule.
  - Because the streams are separate, the drift sequence is the same however long training lasts.
- **Training images**: always come in the same order. A participant's answers only determine how many of them they see.
- **Exact image order**: viewable on the admin page under **Planned image order**.
- **Logged for reproduction**: everything needed to reproduce a session is saved with it — the seed, the algorithm, both schedules, the configuration, and hashes of the PCA file and the face list.

## 8. Researcher-adjustable settings (admin page)

| Setting | Default | Constraint |
| --- | --- | --- |
| Center A initial angle | 0° | Also the start of the drift |
| Center B initial angle | 180° | Always A + 180° |
| Rotation per drift trial | 1° | > 0 and ≤ 360 |
| Number of drift trials | 360 | Set automatically to 360 ÷ rotation (one full rotation); can still be edited |

- Saved settings apply to sessions that start afterwards. A session already running keeps the settings it started with.
- If the settings cannot be loaded, the session does not start.

Fixed in code (not adjustable on the admin page):

| Parameter | Value |
| --- | --- |
| Minimum training trials | 20 |
| Accuracy window | 20 |
| Accuracy threshold | 80% |
| Training block size | 4 |
| Feedback duration | 1,000 ms |
| Display scale | 2× |
| Target radius | 0.45 |
| Faces per trial | 1 (the network experiment used clusters of 64) |

## 9. Data collected

One record is saved for every answered trial, sent to the server as soon as the participant answers. The server stores it in `~/exp_data/representation_stability/<sessionId>.jsonl`, outside the web root.

Per-trial fields:
- **Identity and position**: participant ID, session ID, overall and within-phase trial number, phase.
- **Stimulus**: image file and index, true group, the image's PC1/PC2 coordinates and angle, its distance to the group center.
- **Centers**: the nominal and actual center angles and coordinates used for that trial.
- **Response**: the four-level answer, the binary answer, correct, response time, whether feedback was shown, timestamp.
- **Training only**: window accuracy and whether the criterion was passed.
- **Loading**: number of failed image loads before the face was shown.

Session metadata:
- Consent given and its time, start and end times.
- Training trials completed, and the trial at which the criterion was met.
- Drift trials completed.
- Completion status: `complete`, `in_progress` or `interrupted`.
- Seed and randomization state, the effective configuration, PCA and face-list hashes.

**Export**:
- The researcher downloads CSV files from the password-protected admin page (`admin.html`), or runs `npm run data:csv` on the `.jsonl` files.
- Column definitions: [README.md](README.md#data).

## 10. Interruptions and errors

- **Image fails to load**: an error with `Try again` is shown. No record is created, the timer does not start, and the centers do not move. The same face is retried.
- **Save fails**: retried automatically up to 3 times, then an error with `Try again` is shown. The experiment does not advance until the answer is saved, and the same record is re-sent.
- **Page reload or browser closed**: the session keeps everything saved so far and is marked `interrupted`. A reload starts a new session from consent, with the same participant ID. Interrupted sessions are never overwritten.
- **Double clicks**: ignored. Only the first click on a trial is recorded.

## 11. Outstanding items before data collection

1. Replace the consent placeholders with ethics-approved text, including the ethics approval number, duration, compensation and contact details ([src/content/consent.js](src/content/consent.js)).
2. Set the Prolific completion URL (`completion.prolificCompletionUrl` in [src/config/experimentConfig.js](src/config/experimentConfig.js)).
3. Decide and document:
   - sample size,
   - inclusion and exclusion criteria (for example, a cap on training length, attention checks, or minimum response times),
   - the analysis plan.
4. Confirm the defaults that were never set in the original specification: 1,000 ms feedback, blocks of 4 training trials, and 2× display scale.
5. Run a pilot to estimate session duration. Training length varies between participants, and drift adds 360 trials.
6. Before launch, check that the settings saved on the admin page match the intended design.
