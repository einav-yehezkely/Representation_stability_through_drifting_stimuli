// Experiment engine: phase progression and trial loop (PRD §4-§9).
//
// The engine knows nothing about the DOM or concrete storage. It receives by injection:
//   presenter: { show(imageUrl) -> Promise<onsetMs>, disableResponses(), clear() }
//              show() resolves with the performance.now() onset of the painted image and enables
//              the response controls at that same moment; it rejects on image load failure.
//   storage:   the storage contract (startSession / saveTrial / completeSession [+ optional]).
//   clock:     { sleep(ms) -> Promise, wallNow() -> Date }
//
// Trial loop order (PRD §7): snapshot centers -> select group -> select one face -> load and
// display (onset) -> accept one response -> save record -> (training: feedback, criterion |
// drift: rotate centers) -> next trial.

import { experimentConfig } from "../config/experimentConfig.js";
import { selectStimulus } from "../scientific/stimulusSelector.js";
import { angleDeg, initialCenters } from "../scientific/trajectory.js";
import { DriftState } from "./driftPhase.js";
import { isCorrect, RESPONSES, toBinary } from "./responseCoding.js";
import {
  createDriftSchedule,
  createRngFactory,
  RNG_ALGORITHM,
  RNG_ALGORITHM_VERSION,
  TrainingSchedule,
} from "./seededRandomization.js";
import { COMPLETION_STATUS } from "./session.js";
import { evaluateCriterion, trainingSnapshot } from "./trainingPhase.js";

export const SCREENS = Object.freeze({
  CONSENT: "consent",
  INSTRUCTIONS: "instructions",
  TRIAL: "trial",
  TRANSITION: "transition",
  COMPLETING: "completing",
  COMPLETE: "complete",
  ERROR: "error",
});

export class ImageLoadError extends Error {
  constructor(imageUrl, cause) {
    super(`The image could not be loaded: ${imageUrl}`);
    this.name = "ImageLoadError";
    this.cause = cause;
  }
}

const realClock = {
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  wallNow: () => new Date(),
};

export class ExperimentEngine {
  constructor({
    config = experimentConfig,
    developmentMode = false,
    developmentOverrides = {},
    pca,
    storage,
    presenter,
    sessionId,
    sessionSeed,
    identity,
    clock = realClock,
  }) {
    this.config = config;
    this.developmentMode = developmentMode;
    this.developmentOverrides = developmentOverrides;
    this.pca = pca;
    this.storage = storage;
    this.presenter = presenter;
    this.clock = clock;
    this.listeners = new Set();

    const rngFactory = createRngFactory(sessionSeed);
    this.trainingRng = rngFactory("trainingSchedule");
    this.driftRng = rngFactory("driftSchedule");
    this.trainingSchedule = new TrainingSchedule(config.randomization.trainingBlockSize, this.trainingRng);
    this.driftSchedule = createDriftSchedule(config.drift.numberOfTrials, this.driftRng);

    this.initial = initialCenters(pca, config);
    this.drift = new DriftState(this.initial, config.drift);
    this.usedTrainingIndices = new Set();
    this.usedDriftIndices = new Set();
    this.records = [];
    this.pending = null; // trial selected but not yet answered
    this.accepting = false;
    this.onset = null;
    this.criterion = { completed: 0, eligible: false, accuracy: null, passed: false };
    this.criterionMetAtTrial = null;
    this.sessionStarted = false;
    this.finished = false;

    this.meta = {
      sessionId,
      participantId: identity.participantId,
      participantIdSource: identity.participantIdSource,
      experimentVersion: config.experimentVersion,
      schemaVersion: config.schemaVersion,
      sessionStartTime: clock.wallNow().toISOString(),
      sessionEndTime: null,
      consentGiven: false,
      consentTimestamp: null,
      consentBypassed: false,
      developmentMode,
      developmentOverrides,
      effectiveConfig: config,
      sessionSeed,
      rngAlgorithm: RNG_ALGORITHM,
      rngAlgorithmVersion: RNG_ALGORITHM_VERSION,
      pcaSource: pca.source,
      pcaSha256: pca.sha256,
      facesListSha256: pca.manifest?.facesListSha256 ?? null,
      assetsManifest: pca.manifest ?? null,
      initialCenters: {
        baseImageId: this.initial.baseImageId,
        baseImageIndex: this.initial.baseImageIndex,
        centerA: this.initial.centerA,
        centerB: this.initial.centerB,
        nominalAngleA: this.initial.nominalAngleA,
        nominalAngleB: this.initial.nominalAngleB,
        actualAngleA: angleDeg(...this.initial.centerA),
        actualAngleB: angleDeg(...this.initial.centerB),
      },
      randomization: this.randomizationState(),
      completionStatus: COMPLETION_STATUS.IN_PROGRESS,
      terminationReason: null,
      trainingCompletedTrials: 0,
      driftCompletedTrials: 0,
      trainingCriterionMet: false,
      trainingCriterionMetAtTrial: null,
    };

    this.state = { screen: SCREENS.CONSENT };
  }

  // ---------------------------------------------------------------- state plumbing

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  setState(state) {
    this.state = state;
    for (const listener of this.listeners) listener(state);
  }

  randomizationState() {
    return {
      algorithm: RNG_ALGORITHM,
      algorithmVersion: RNG_ALGORITHM_VERSION,
      sessionSeed: this.trainingRng.sessionSeed,
      streams: [this.trainingRng.state(), this.driftRng.state()],
      trainingBlockSize: this.trainingSchedule.blockSize,
      trainingBlocks: this.trainingSchedule.blocks.map((b) => b.join("")),
      driftSchedule: this.driftSchedule.join(""),
      balancingPolicy:
        "training: shuffled balanced blocks (blockSize/2 A + blockSize/2 B); drift: one shuffled list with floor(N/2) A and floor(N/2) B (+1 random group if N is odd)",
    };
  }

  get trainingRecords() {
    return this.records.filter((r) => r.phase === "training");
  }

  get driftRecords() {
    return this.records.filter((r) => r.phase === "drift");
  }

  // ---------------------------------------------------------------- consent & instructions

  async acceptConsent() {
    if (this.state.screen !== SCREENS.CONSENT) return;
    this.meta.consentGiven = true;
    this.meta.consentTimestamp = this.clock.wallNow().toISOString();
    await this.startSession();
  }

  /** Development only: skip consent, recorded honestly as a bypass (not as consent). */
  async bypassConsent() {
    if (!this.developmentMode) throw new Error("Consent bypass is only available in development mode.");
    if (this.state.screen !== SCREENS.CONSENT) return;
    this.meta.consentGiven = false;
    this.meta.consentBypassed = true;
    await this.startSession();
  }

  async startSession() {
    try {
      await this.storage.startSession({ ...this.meta });
      this.sessionStarted = true;
      this.setState({ screen: SCREENS.INSTRUCTIONS });
    } catch (error) {
      this.showError("storage", error, () => this.startSession());
    }
  }

  async startTraining() {
    if (this.state.screen !== SCREENS.INSTRUCTIONS) return;
    await this.runTrial();
  }

  // ---------------------------------------------------------------- trial loop

  get phase() {
    return this.criterion.passed ? "drift" : "training";
  }

  prepareTrial() {
    const phase = this.phase;
    const snapshot = phase === "training" ? trainingSnapshot(this.initial) : this.drift.snapshot();
    const group = phase === "training" ? this.trainingSchedule.next() : this.driftSchedule[this.drift.completed];
    const excluded = phase === "training" ? this.usedTrainingIndices : this.usedDriftIndices;
    const stimulus = selectStimulus({
      group,
      centers: snapshot,
      pca: this.pca,
      excluded,
      clusterSize: this.config.scientific.clusterSize,
    });
    const phaseTrialNumber = phase === "training" ? this.trainingRecords.length + 1 : this.drift.nextTrialNumber;
    return { phase, phaseTrialNumber, group, snapshot, stimulus, loadFailures: 0 };
  }

  imageUrl(imageId) {
    return `${this.config.assets.facesDirectory}${encodeURIComponent(imageId)}`;
  }

  async runTrial() {
    if (this.finished) return;
    if (!this.pending) this.pending = this.prepareTrial();
    const pending = this.pending;
    this.accepting = false;
    this.onset = null;
    this.setState({ screen: SCREENS.TRIAL, status: "loading", phase: pending.phase, feedback: null });
    let onset;
    try {
      onset = await this.presenter.show(this.imageUrl(pending.stimulus.imageId));
    } catch (error) {
      // A failed load never counts as an answered trial, never starts a timer and never moves
      // centers; the same trial (group and image) is retried.
      pending.loadFailures += 1;
      this.showError("imageLoad", error, () => this.runTrial());
      return;
    }
    if (this.finished || this.pending !== pending) return;
    this.onset = onset;
    this.accepting = true;
    this.setState({ screen: SCREENS.TRIAL, status: "responding", phase: pending.phase, feedback: null });
  }

  /**
   * Accept a participant response. `responseTime` is the performance.now() value captured first
   * thing in the input handler. Only the first response per trial is accepted.
   */
  async respond(response, responseTime) {
    if (!this.accepting || !this.pending) return false;
    if (!RESPONSES.includes(response)) throw new Error(`Unknown response: ${response}`);
    this.accepting = false;
    const responseTimeMs = responseTime - this.onset;
    const timestamp = this.clock.wallNow().toISOString();
    this.presenter.disableResponses();

    const pending = this.pending;
    const binaryResponse = toBinary(response);
    const correct = isCorrect(binaryResponse, pending.group);
    let criterionAfter = null;
    if (pending.phase === "training") {
      criterionAfter = evaluateCriterion([...this.trainingRecords, { correct }], this.config.training);
    }
    const { snapshot, stimulus } = pending;
    const record = {
      participantId: this.meta.participantId,
      sessionId: this.meta.sessionId,
      trialNumber: this.records.length + 1,
      phaseTrialNumber: pending.phaseTrialNumber,
      phase: pending.phase,
      imageId: stimulus.imageId,
      trueGroup: pending.group,
      pc1: stimulus.pc1,
      pc2: stimulus.pc2,
      stimulusAngle: stimulus.stimulusAngle,
      currentAngleA: snapshot.nominalAngleA,
      currentAngleB: snapshot.nominalAngleB,
      response,
      binaryResponse,
      correct,
      responseTimeMs,
      feedbackShown: pending.phase === "training",
      timestamp,
      imageIndex: stimulus.imageIndex,
      distanceToCenter: stimulus.distanceToCenter,
      selectionRule: pending.phase === "training" ? this.config.scientific.trainingSelection : this.config.scientific.driftSelection,
      centerA_pc1: snapshot.centerA[0],
      centerA_pc2: snapshot.centerA[1],
      centerA_actualAngle: angleDeg(snapshot.centerA[0], snapshot.centerA[1]),
      centerB_pc1: snapshot.centerB[0],
      centerB_pc2: snapshot.centerB[1],
      centerB_actualAngle: angleDeg(snapshot.centerB[0], snapshot.centerB[1]),
      windowAccuracy: criterionAfter ? criterionAfter.accuracy : null,
      criterionPassed: criterionAfter ? criterionAfter.passed : null,
      imageLoadFailures: pending.loadFailures,
    };
    await this.persistAndAdvance(record, criterionAfter);
    return true;
  }

  async persistAndAdvance(record, criterionAfter) {
    this.setState({ screen: SCREENS.TRIAL, status: "saving", phase: record.phase, feedback: null });
    try {
      await this.storage.saveTrial(record);
    } catch (error) {
      // Not saved: do not advance, keep the record for a retry with the same stable key.
      this.showError("storage", error, () => this.persistAndAdvance(record, criterionAfter));
      return;
    }
    if (this.finished) return;
    this.records.push(Object.freeze(record));
    this.pending = null;

    if (record.phase === "training") {
      this.usedTrainingIndices.add(record.imageIndex);
      this.criterion = criterionAfter;
      const text = record.correct ? this.config.ui.feedbackText.correct : this.config.ui.feedbackText.incorrect;
      this.setState({ screen: SCREENS.TRIAL, status: "feedback", phase: "training", feedback: text, correct: record.correct });
      await this.clock.sleep(this.config.ui.feedbackDurationMs);
      if (this.finished) return;
      if (criterionAfter.passed) {
        this.criterionMetAtTrial = record.trialNumber;
        this.presenter.clear();
        this.setState({ screen: SCREENS.TRANSITION });
        return;
      }
      await this.runTrial();
      return;
    }

    // Drift: move both centers only after the response (PRD §7 step 6).
    this.usedDriftIndices.add(record.imageIndex);
    this.drift.advanceAfterResponse();
    if (this.drift.done) {
      await this.complete(COMPLETION_STATUS.COMPLETE, "completed_all_drift_trials");
      return;
    }
    await this.runTrial();
  }

  async continueToDrift() {
    if (this.state.screen !== SCREENS.TRANSITION) return;
    await this.runTrial();
  }

  // ---------------------------------------------------------------- completion

  completionMetadata(status, reason) {
    return {
      sessionId: this.meta.sessionId,
      sessionEndTime: this.clock.wallNow().toISOString(),
      completionStatus: status,
      terminationReason: reason,
      trainingCompletedTrials: this.trainingRecords.length,
      driftCompletedTrials: this.driftRecords.length,
      trainingCriterionMet: this.criterion.passed,
      trainingCriterionMetAtTrial: this.criterionMetAtTrial,
      randomization: this.randomizationState(),
    };
  }

  async complete(status, reason) {
    this.finished = true;
    this.accepting = false;
    this.presenter.clear();
    const completion = this.completionMetadata(status, reason);
    Object.assign(this.meta, completion);
    this.setState({ screen: SCREENS.COMPLETING, completionStatus: status });
    try {
      const result = await this.storage.completeSession(completion);
      this.setState({ screen: SCREENS.COMPLETE, completionStatus: status, ...result });
    } catch (error) {
      this.showError("storage", error, () => this.complete(status, reason));
    }
  }

  /** Re-export after completion (manual download / retry). */
  async downloadAgain() {
    if (typeof this.storage.exportSession !== "function") return null;
    const result = await this.storage.exportSession(this.meta.sessionId, {
      incomplete: this.meta.completionStatus !== COMPLETION_STATUS.COMPLETE,
    });
    if (this.state.screen === SCREENS.COMPLETE) this.setState({ ...this.state, ...result });
    return result;
  }

  // ---------------------------------------------------------------- development controls

  /** Development only: stop now and save the session as incomplete. */
  async finishEarly() {
    if (!this.developmentMode) throw new Error("Finish early is only available in development mode.");
    if (!this.sessionStarted || this.finished) return;
    await this.complete(COMPLETION_STATUS.DEV_EARLY_FINISH, "development_early_finish");
  }

  /** Development only: download the data saved so far, marked incomplete. */
  async exportPartial() {
    if (!this.developmentMode) throw new Error("Partial export is only available in development mode.");
    if (!this.sessionStarted || typeof this.storage.exportSession !== "function") return null;
    return this.storage.exportSession(this.meta.sessionId, { incomplete: true });
  }

  /** Debug information for the development panel (never rendered in participant mode). */
  debugInfo() {
    const pending = this.pending;
    const snapshot = pending?.snapshot ?? (this.phase === "drift" ? this.drift.snapshot() : trainingSnapshot(this.initial));
    return {
      screen: this.state.screen,
      phase: this.phase,
      trialNumber: this.records.length + (pending ? 1 : 0),
      phaseTrialNumber: pending?.phaseTrialNumber ?? null,
      group: pending?.group ?? null,
      imageId: pending?.stimulus.imageId ?? null,
      stimulusAngle: pending?.stimulus.stimulusAngle ?? null,
      distanceToCenter: pending?.stimulus.distanceToCenter ?? null,
      nominalAngleA: snapshot.nominalAngleA,
      nominalAngleB: snapshot.nominalAngleB,
      actualAngleA: angleDeg(...snapshot.centerA),
      actualAngleB: angleDeg(...snapshot.centerB),
      centerA: snapshot.centerA,
      centerB: snapshot.centerB,
      trainingCompleted: this.trainingRecords.length,
      windowAccuracy: this.criterion.accuracy,
      criterionPassed: this.criterion.passed,
      driftCompleted: this.driftRecords.length,
      sessionSeed: this.meta.sessionSeed,
    };
  }

  // ---------------------------------------------------------------- errors

  showError(kind, error, retry) {
    this.setState({ screen: SCREENS.ERROR, kind, message: error?.message ?? String(error), code: error?.code, retry });
  }

  async retry() {
    if (this.state.screen === SCREENS.ERROR && typeof this.state.retry === "function") await this.state.retry();
  }
}
