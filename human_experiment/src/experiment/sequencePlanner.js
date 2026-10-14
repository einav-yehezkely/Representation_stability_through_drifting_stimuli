// Pure lookahead of the stimuli the engine will present. It uses the engine's own schedules and
// stimulus selector on copies of the state, so it never changes what the participant sees.
//
// Stimulus selection is deterministic (decisions D2, D4, D5) and the group schedules depend only
// on the session seed, so the image order is known in advance:
//   * training: the order is fixed by the seed; participant responses only decide how long
//     training lasts.
//   * drift: the order is fixed by the seed and does not depend on training at all.
// Used to preload upcoming images (experimentEngine.preloadUpcoming) and by the admin page to
// preview the image order of a seed.

import { selectStimulus } from "../scientific/stimulusSelector.js";
import { initialCenters } from "../scientific/trajectory.js";
import { DriftState } from "./driftPhase.js";
import { createDriftSchedule, createRngFactory, TrainingSchedule } from "./seededRandomization.js";
import { trainingSnapshot } from "./trainingPhase.js";

/**
 * Training stimuli for the given groups, in order, at the fixed training centers.
 * @param {{groups:string[], initial:object, pca:object, excluded:Set<number>, clusterSize:number, firstTrialNumber?:number}} args
 */
export function planTrainingStimuli({ groups, initial, pca, excluded, clusterSize, firstTrialNumber = 1 }) {
  const used = new Set(excluded);
  const snapshot = trainingSnapshot(initial);
  return groups.map((group, i) => {
    const stimulus = selectStimulus({ group, centers: snapshot, pca, excluded: used, clusterSize });
    used.add(stimulus.imageIndex);
    return { phase: "training", phaseTrialNumber: firstTrialNumber + i, group, snapshot, stimulus };
  });
}

/**
 * Up to `count` drift stimuli from the given drift state (which is copied, not advanced).
 * @param {{drift:DriftState, schedule:string[], count:number, pca:object, excluded:Set<number>, clusterSize:number}} args
 */
export function planDriftStimuli({ drift, schedule, count, pca, excluded, clusterSize }) {
  const state = drift.clone();
  const used = new Set(excluded);
  const planned = [];
  while (planned.length < count && !state.done) {
    const snapshot = state.snapshot();
    const group = schedule[state.completed];
    const stimulus = selectStimulus({ group, centers: snapshot, pca, excluded: used, clusterSize });
    used.add(stimulus.imageIndex);
    planned.push({ phase: "drift", phaseTrialNumber: state.nextTrialNumber, group, snapshot, stimulus });
    state.advanceAfterResponse();
  }
  return planned;
}

/**
 * The complete image order of a session with this seed and configuration, assuming training
 * lasts `trainingTrials` trials. Mirrors the construction in ExperimentEngine.
 */
export function planSession({ pca, config, sessionSeed, trainingTrials }) {
  const rngFactory = createRngFactory(sessionSeed);
  const trainingSchedule = new TrainingSchedule(config.randomization.trainingBlockSize, rngFactory("trainingSchedule"));
  const driftSchedule = createDriftSchedule(config.drift.numberOfTrials, rngFactory("driftSchedule"));
  const initial = initialCenters(pca, config);
  const clusterSize = config.scientific.clusterSize;
  return {
    training: planTrainingStimuli({ groups: trainingSchedule.peek(trainingTrials), initial, pca, excluded: new Set(), clusterSize }),
    drift: planDriftStimuli({
      drift: new DriftState(initial, config.drift),
      schedule: driftSchedule,
      count: config.drift.numberOfTrials,
      pca,
      excluded: new Set(),
      clusterSize,
    }),
  };
}
