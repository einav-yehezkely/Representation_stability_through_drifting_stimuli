// Training phase: fixed centers, feedback, sliding-window learning criterion (PRD §6).

/**
 * Evaluate the learning criterion over completed training trials (including the one just
 * completed). Sliding window over the LAST `accuracyWindow` trials, never cumulative accuracy,
 * and never on an incomplete window.
 *
 * @param {{correct:boolean}[]} history completed training trials in order
 */
export function evaluateCriterion(history, { minTrials, accuracyWindow, accuracyThreshold }) {
  const completed = history.length;
  const eligible = completed >= minTrials && completed >= accuracyWindow;
  if (!eligible) return { completed, eligible, accuracy: null, passed: false };
  const window = history.slice(-accuracyWindow);
  const accuracy = window.filter((trial) => trial.correct).length / accuracyWindow;
  return { completed, eligible, accuracy, passed: accuracy >= accuracyThreshold };
}

/** Training centers never move: the snapshot is the initial centers for every training trial. */
export function trainingSnapshot(initial) {
  return {
    centerA: [initial.centerA[0], initial.centerA[1]],
    centerB: [initial.centerB[0], initial.centerB[1]],
    nominalAngleA: initial.nominalAngleA,
    nominalAngleB: initial.nominalAngleB,
  };
}
