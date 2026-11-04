// Experiment engine: session flow and trial loop.
//
// The engine knows nothing about the DOM or concrete storage. It receives by injection:
//   presenter: { load(urls), showSample(url) -> onsetMs, hideSample() -> offsetMs,
//                showPair(leftUrl, rightUrl) -> onsetMs, disableResponses(), clear(), [preload(urls)] }
//              load() fetches and decodes the images of a trial and rejects with ImageLoadError;
//              it runs before the sample appears, so a slow image can never stretch the sample,
//              the blank screen or the response time. showPair() enables the responses at the
//              moment it takes the onset. The optional preload() receives the URLs of the next
//              images; it never changes what is shown or when.
//   storage:   the storage contract (startSession / saveTrial / completeSession [+ optional]).
//   clock:     { sleep(ms) -> Promise, wallNow() -> Date }
//
// Trial loop: load images -> sample (timing.sampleDurationMs) -> blank (timing.blankDurationMs) ->
// two choices (onset) -> accept one response -> empty screen (timing.interTrialDurationMs, counted
// from the click) -> next trial. The answer is saved and the next images load during that empty
// screen, so neither lengthens it unless it takes longer. The trial sequence comes from
// sequencePlanner.planSession; the centers of a trial are those of its position in the sequence,
// so they advance only when an answer has been saved.

import { experimentConfig } from "../config/experimentConfig.js";
import { angleDeg } from "../scientific/trajectory.js";
import { chosenGroup, isCorrect, RESPONSES } from "./responseCoding.js";
import { RNG_ALGORITHM, RNG_ALGORITHM_VERSION } from "./seededRandomization.js";
import { planSession } from "./sequencePlanner.js";
import { COMPLETION_STATUS } from "./session.js";

export const SCREENS = Object.freeze({
  CONSENT: "consent",
  INSTRUCTIONS: "instructions",
  TRIAL: "trial",
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

/** Record fields describing one image; prefix is "sample", "optionA" or "optionB". */
function stimulusFields(prefix, stimulus) {
  return {
    [`${prefix}ImageId`]: stimulus.imageId,
    [`${prefix}ImageIndex`]: stimulus.imageIndex,
    [`${prefix}Pc1`]: stimulus.pc1,
    [`${prefix}Pc2`]: stimulus.pc2,
    [`${prefix}Angle`]: stimulus.stimulusAngle,
    [`${prefix}DistanceToCenter`]: stimulus.distanceToCenter,
  };
}

export class ExperimentEngine {
  constructor({
    config = experimentConfig,
    developmentMode = false,
    developmentOverrides = {},
    experimentSettings = null,
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

    this.plan = planSession({ pca, config, sessionSeed });
    this.initial = this.plan.initial;
    this.sessionSeed = sessionSeed;
    this.records = [];
    this.pending = null; // current trial, not yet answered
    this.accepting = false;
    this.onset = null;
    this.interTrial = null; // running pause after the last click
    this.lastResponseTime = null;
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
      // Settings saved on the admin page that this session applied ({ values, updatedAt, source });
      // their values are also part of effectiveConfig.
      experimentSettings,
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
      completedTrials: 0,
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
      sessionSeed: this.sessionSeed,
      streams: this.plan.streams,
      sampleSchedule: this.plan.sampleSchedule.join(""),
      leftSchedule: this.plan.leftSchedule.join(""),
      balancingPolicy:
        "sampleSchedule: trial 1 is A, the remaining trials are one shuffled list with floor((N-1)/2) A and B (+1 random group if N-1 is odd); leftSchedule: one shuffled list with floor(N/2) A and B (+1 random group if N is odd)",
    };
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
      this.preloadUpcoming(0);
    } catch (error) {
      this.showError("storage", error, () => this.startSession());
    }
  }

  async startExperiment() {
    if (this.state.screen !== SCREENS.INSTRUCTIONS) return;
    await this.runTrial();
  }

  // ---------------------------------------------------------------- trial loop

  imageUrl(imageId) {
    return `${this.config.assets.facesDirectory}${encodeURIComponent(imageId)}`;
  }

  /** URLs of the three images of a trial: sample, then the choices. */
  trialUrls(trial) {
    return [trial.sample, trial.options.A, trial.options.B].map((s) => this.imageUrl(s.imageId));
  }

  async runTrial() {
    if (this.finished) return;
    this.pending ??= { trial: this.plan.trials[this.records.length], loadFailures: 0 };
    const pending = this.pending;
    const { trial } = pending;
    const active = () => !this.finished && this.pending === pending;
    const { sampleDurationMs, blankDurationMs } = this.config.timing;
    this.accepting = false;
    this.onset = null;
    this.setState({ screen: SCREENS.TRIAL, status: "loading" });
    try {
      await this.presenter.load(this.trialUrls(trial));
    } catch (error) {
      // A failed load never counts as an answered trial and never moves the centers; the same
      // trial is retried.
      pending.loadFailures += 1;
      this.showError("imageLoad", error, () => this.runTrial());
      return;
    }
    await this.interTrial;
    this.interTrial = null;
    if (!active()) return;

    const sampleOnset = await this.presenter.showSample(this.imageUrl(trial.sample.imageId));
    if (!active()) return;
    if (this.lastResponseTime !== null) pending.interTrialDurationMs = sampleOnset - this.lastResponseTime;
    this.setState({ screen: SCREENS.TRIAL, status: "sample" });
    await this.clock.sleep(sampleDurationMs);
    if (!active()) return;

    const sampleOffset = await this.presenter.hideSample();
    if (!active()) return;
    this.setState({ screen: SCREENS.TRIAL, status: "blank" });
    await this.clock.sleep(blankDurationMs);
    if (!active()) return;

    const [left, right] = [trial.leftGroup, trial.rightGroup].map((group) => this.imageUrl(trial.options[group].imageId));
    const onset = await this.presenter.showPair(left, right);
    if (!active()) return;
    pending.sampleDurationMs = sampleOffset - sampleOnset;
    pending.blankDurationMs = onset - sampleOffset;
    this.onset = onset;
    this.accepting = true;
    this.setState({ screen: SCREENS.TRIAL, status: "responding" });
  }

  /**
   * Accept a participant response ("left" or "right"). `responseTime` is the performance.now()
   * value captured first thing in the input handler. Only the first response per trial counts.
   */
  async respond(response, responseTime) {
    if (!this.accepting || !this.pending) return false;
    if (!RESPONSES.includes(response)) throw new Error(`Unknown response: ${response}`);
    this.accepting = false;
    const responseTimeMs = responseTime - this.onset;
    const timestamp = this.clock.wallNow().toISOString();
    this.presenter.disableResponses();
    this.presenter.clear();
    this.lastResponseTime = responseTime;
    this.interTrial = this.clock.sleep(this.config.timing.interTrialDurationMs);

    const { trial, loadFailures, sampleDurationMs, blankDurationMs, interTrialDurationMs = null } = this.pending;
    const chosen = chosenGroup(response, trial);
    // Start fetching the next images now, so they load while this answer is being saved.
    this.preloadUpcoming(trial.trialNumber);
    const { snapshot } = trial;
    const record = {
      participantId: this.meta.participantId,
      sessionId: this.meta.sessionId,
      trialNumber: trial.trialNumber,
      sampleGroup: trial.sampleGroup,
      sampleSourceTrial: trial.sampleSourceTrial,
      ...stimulusFields("sample", trial.sample),
      ...stimulusFields("optionA", trial.options.A),
      ...stimulusFields("optionB", trial.options.B),
      leftGroup: trial.leftGroup,
      currentAngleA: snapshot.nominalAngleA,
      currentAngleB: snapshot.nominalAngleB,
      response,
      chosenGroup: chosen,
      chosenImageId: trial.options[chosen].imageId,
      correct: isCorrect(chosen, trial.sampleGroup),
      responseTimeMs,
      sampleDurationMs,
      blankDurationMs,
      interTrialDurationMs,
      timestamp,
      selectionRule: this.config.scientific.selection,
      centerA_pc1: snapshot.centerA[0],
      centerA_pc2: snapshot.centerA[1],
      centerA_actualAngle: angleDeg(snapshot.centerA[0], snapshot.centerA[1]),
      centerB_pc1: snapshot.centerB[0],
      centerB_pc2: snapshot.centerB[1],
      centerB_actualAngle: angleDeg(snapshot.centerB[0], snapshot.centerB[1]),
      imageLoadFailures: loadFailures,
    };
    await this.persistAndAdvance(record);
    return true;
  }

  async persistAndAdvance(record) {
    this.setState({ screen: SCREENS.TRIAL, status: "saving" });
    try {
      await this.storage.saveTrial(record);
    } catch (error) {
      // Not saved: do not advance, keep the record for a retry with the same stable key.
      this.showError("storage", error, () => this.persistAndAdvance(record));
      return;
    }
    if (this.finished) return;
    this.records.push(Object.freeze(record));
    this.pending = null;
    if (this.records.length >= this.config.drift.numberOfTrials) {
      await this.complete(COMPLETION_STATUS.COMPLETE, "completed_all_trials");
      return;
    }
    await this.runTrial();
  }

  // ---------------------------------------------------------------- preloading

  /** Hand the URLs of the trials after `answered` to the presenter (an optimization only). */
  preloadUpcoming(answered) {
    const count = this.config.ui.preloadCount ?? 0;
    if (!count || typeof this.presenter?.preload !== "function") return;
    const urls = this.plan.trials.slice(answered, answered + count).flatMap((trial) => this.trialUrls(trial));
    this.presenter.preload([...new Set(urls)]);
  }

  // ---------------------------------------------------------------- completion

  completionMetadata(status, reason) {
    return {
      sessionId: this.meta.sessionId,
      sessionEndTime: this.clock.wallNow().toISOString(),
      completionStatus: status,
      terminationReason: reason,
      completedTrials: this.records.length,
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
    const trial = this.pending?.trial ?? null;
    const snapshot = (trial ?? this.plan.trials[Math.min(this.records.length, this.plan.trials.length - 1)]).snapshot;
    return {
      screen: this.state.screen,
      status: this.state.status ?? null,
      trialNumber: trial?.trialNumber ?? null,
      sampleGroup: trial?.sampleGroup ?? null,
      sampleImageId: trial?.sample.imageId ?? null,
      sampleSourceTrial: trial?.sampleSourceTrial ?? null,
      sampleAngle: trial?.sample.stimulusAngle ?? null,
      optionAImageId: trial?.options.A.imageId ?? null,
      optionBImageId: trial?.options.B.imageId ?? null,
      leftGroup: trial?.leftGroup ?? null,
      nominalAngleA: snapshot.nominalAngleA,
      nominalAngleB: snapshot.nominalAngleB,
      actualAngleA: angleDeg(...snapshot.centerA),
      actualAngleB: angleDeg(...snapshot.centerB),
      centerA: snapshot.centerA,
      centerB: snapshot.centerB,
      completedTrials: this.records.length,
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
