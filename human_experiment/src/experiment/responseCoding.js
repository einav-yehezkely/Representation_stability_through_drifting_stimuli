// Response coding (PRD §5). Original four-level responses are preserved verbatim; the binary
// response is derived separately. Confidence does not affect correctness.

export const RESPONSES = Object.freeze(["A", "Probably A", "Probably B", "B"]);

const BINARY = Object.freeze({
  A: "A",
  "Probably A": "A",
  "Probably B": "B",
  B: "B",
});

export function toBinary(response) {
  const binary = BINARY[response];
  if (!binary) throw new Error(`Unknown response: ${response}`);
  return binary;
}

export function isCorrect(binaryResponse, trueGroup) {
  return binaryResponse === trueGroup;
}
