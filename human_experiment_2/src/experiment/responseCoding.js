// Response coding. The participant clicks one of the two faces; the side is stored verbatim and
// the group of the chosen face is derived from the side assignment of the trial.

export const RESPONSES = Object.freeze(["left", "right"]);

export function chosenGroup(response, trial) {
  if (response === "left") return trial.leftGroup;
  if (response === "right") return trial.rightGroup;
  throw new Error(`Unknown response: ${response}`);
}

/** Correct: the chosen face belongs to the same group as the sample. */
export function isCorrect(group, sampleGroup) {
  return group === sampleGroup;
}
