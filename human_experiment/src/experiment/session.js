// Participant and session identity, session metadata (PRD §11).

export const COMPLETION_STATUS = Object.freeze({
  IN_PROGRESS: "in_progress",
  COMPLETE: "complete",
  DEV_EARLY_FINISH: "incomplete_dev_early_finish",
  INTERRUPTED: "interrupted",
});

function randomHex(bytes, cryptoImpl) {
  const buffer = new Uint8Array(bytes);
  cryptoImpl.getRandomValues(buffer);
  return [...buffer].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Participant identifier: a non-empty PROLIFIC_PID URL parameter when present, otherwise a
 * generated anonymous identifier. Only the parameter is read; the URL itself is never stored.
 */
export function resolveParticipantId(search, cryptoImpl = globalThis.crypto) {
  const params = new URLSearchParams(search ?? "");
  const pid = (params.get("PROLIFIC_PID") ?? "").trim();
  if (pid) return { participantId: pid, participantIdSource: "prolific_url" };
  return { participantId: `anon-${randomHex(8, cryptoImpl)}`, participantIdSource: "generated_anonymous" };
}

export function createSessionId(cryptoImpl = globalThis.crypto) {
  return cryptoImpl.randomUUID();
}

/** ISO 8601 UTC wall-clock timestamp. */
export function utcNow(now = () => new Date()) {
  return now().toISOString();
}
