// Center drift: both centers rotate together after every response.
//
// Movement order (source main loop, classify_rotation_resnet50.py lines 1512-1784): select and
// present at the current centers, then rotate both centers by degreesPerTrial with rotate_vector.
// The first trial therefore uses the initial centers.

import { nominalDriftAngle, rotateVector } from "../scientific/trajectory.js";

export class DriftState {
  constructor(initial, { degreesPerTrial, numberOfTrials }) {
    this.initialAngleA = initial.nominalAngleA;
    this.initialAngleB = initial.nominalAngleB;
    this.centerA = [initial.centerA[0], initial.centerA[1]];
    this.centerB = [initial.centerB[0], initial.centerB[1]];
    this.degreesPerTrial = degreesPerTrial;
    this.numberOfTrials = numberOfTrials;
    this.completed = 0; // answered trials
  }

  /** One-based index of the trial about to be displayed. */
  get nextTrialNumber() {
    return this.completed + 1;
  }

  get done() {
    return this.completed >= this.numberOfTrials;
  }

  /** Centers that generate the two choices of a trial (copied; later rotation cannot alter it). */
  snapshot() {
    const k = this.nextTrialNumber;
    return {
      centerA: [this.centerA[0], this.centerA[1]],
      centerB: [this.centerB[0], this.centerB[1]],
      nominalAngleA: nominalDriftAngle(this.initialAngleA, k, this.degreesPerTrial),
      nominalAngleB: nominalDriftAngle(this.initialAngleB, k, this.degreesPerTrial),
    };
  }

  /** Moves both centers to their position for the next trial. */
  advanceAfterResponse() {
    this.centerA = rotateVector(this.centerA, this.degreesPerTrial);
    this.centerB = rotateVector(this.centerB, this.degreesPerTrial);
    this.completed += 1;
  }
}
