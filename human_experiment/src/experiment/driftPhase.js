// Drift phase: both centers rotate together after every response (PRD §7).
//
// Movement order (source main loop, classify_rotation_resnet50.py lines 1512-1784): select and
// present at the current centers, then rotate both centers by degreesPerTrial with rotate_vector.
// The first drift trial therefore uses the initial (training) centers.

import { nominalDriftAngle, rotateVector } from "../scientific/trajectory.js";

export class DriftState {
  constructor(initial, { degreesPerTrial, numberOfTrials }) {
    this.initialAngleA = initial.nominalAngleA;
    this.initialAngleB = initial.nominalAngleB;
    this.centerA = [initial.centerA[0], initial.centerA[1]];
    this.centerB = [initial.centerB[0], initial.centerB[1]];
    this.degreesPerTrial = degreesPerTrial;
    this.numberOfTrials = numberOfTrials;
    this.completed = 0; // answered drift trials
  }

  /** One-based index of the drift trial about to be displayed. */
  get nextTrialNumber() {
    return this.completed + 1;
  }

  get done() {
    return this.completed >= this.numberOfTrials;
  }

  /** Centers that generate the displayed face (copied; later rotation cannot alter it). */
  snapshot() {
    const k = this.nextTrialNumber;
    return {
      centerA: [this.centerA[0], this.centerA[1]],
      centerB: [this.centerB[0], this.centerB[1]],
      nominalAngleA: nominalDriftAngle(this.initialAngleA, k, this.degreesPerTrial),
      nominalAngleB: nominalDriftAngle(this.initialAngleB, k, this.degreesPerTrial),
    };
  }

  /** Called only after an accepted, saved response. */
  advanceAfterResponse() {
    this.centerA = rotateVector(this.centerA, this.degreesPerTrial);
    this.centerB = rotateVector(this.centerB, this.degreesPerTrial);
    this.completed += 1;
  }
}
