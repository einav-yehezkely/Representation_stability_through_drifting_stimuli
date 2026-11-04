// The complete trial sequence of a session.
//
// A trial shows a sample face, then two choices: one face from each group, selected at the current
// centers. Continuity: the sample of every trial after the first is one of the two choices of the
// previous trial, namely the one of the group the sample schedule names for this trial. The sample
// of trial 1 is the image nearest to center A. Which choice continues never depends on the answer.
//
// Nothing here depends on the participant, so the sequence is fixed by the seed and the
// configuration. Used by the engine, and by the admin page to preview the image order.

import { selectStimulus } from "../scientific/stimulusSelector.js";
import { initialCenters } from "../scientific/trajectory.js";
import { DriftState } from "./driftPhase.js";
import { createBalancedSchedule, createRngFactory } from "./seededRandomization.js";

export const FIRST_SAMPLE_GROUP = "A";

const otherGroup = (group) => (group === "A" ? "B" : "A");

/**
 * @param {{pca:object, config:object, sessionSeed:number, limit?:number}} args
 *   limit: plan only the first `limit` trials (a trial never depends on later ones).
 * @returns {{trials:object[], sampleSchedule:string[], leftSchedule:string[], initial:object, streams:object[]}}
 */
export function planSession({ pca, config, sessionSeed, limit = config.drift.numberOfTrials }) {
  const rngFactory = createRngFactory(sessionSeed);
  const sampleRng = rngFactory("sampleSchedule");
  const sideRng = rngFactory("sideSchedule");
  const n = config.drift.numberOfTrials;
  const sampleSchedule = [FIRST_SAMPLE_GROUP, ...createBalancedSchedule(n - 1, sampleRng)];
  const leftSchedule = createBalancedSchedule(n, sideRng);

  const initial = initialCenters(pca, config);
  const drift = new DriftState(initial, config.drift);
  const { clusterSize } = config.scientific;
  const used = new Set();
  const pick = (group, centers) => {
    const stimulus = selectStimulus({ group, centers, pca, excluded: used, clusterSize });
    used.add(stimulus.imageIndex);
    return stimulus;
  };

  const trials = [];
  while (trials.length < limit && !drift.done) {
    const snapshot = drift.snapshot();
    const trialNumber = drift.nextTrialNumber;
    const sampleGroup = sampleSchedule[trialNumber - 1];
    const previous = trials.at(-1);
    const sample = previous ? previous.options[sampleGroup] : pick(sampleGroup, snapshot);
    const options = { A: pick("A", snapshot), B: pick("B", snapshot) };
    const leftGroup = leftSchedule[trialNumber - 1];
    trials.push({
      trialNumber,
      sampleGroup,
      sample,
      sampleSourceTrial: previous ? previous.trialNumber : null,
      options,
      leftGroup,
      rightGroup: otherGroup(leftGroup),
      snapshot,
    });
    drift.advanceAfterResponse();
  }
  return { trials, sampleSchedule, leftSchedule, initial, streams: [sampleRng.state(), sideRng.state()] };
}
