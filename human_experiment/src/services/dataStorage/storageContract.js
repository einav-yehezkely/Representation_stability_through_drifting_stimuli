// Storage contract (PRD §13). Scientific and UI logic depend only on this interface.
//
// Required methods (all async):
//
//   startSession(sessionMetadata) -> { ok: true }
//       Persist a new session. Must never overwrite or discard another session.
//
//   saveTrial(trialRecord) -> { ok: true, duplicate?: boolean }
//       Persist one immutable answered-trial record. Idempotent on the stable key
//       `${sessionId}:${trialNumber}`: a retry with an existing key is a no-op (duplicate: true).
//       Must not trigger a download.
//
//   completeSession(completionMetadata) -> { ok: true, exported: boolean, exportName?: string, exportError?: string }
//       Persist the final session metadata, then hand the data to the researcher (for the pilot
//       adapter: download `<sessionId>.csv`). `exported: false` means the data is saved but the
//       hand-off did not happen (e.g. a blocked download); the UI then offers a manual retry.
//
// Optional methods (used by recovery/development/review UI; not used by phase logic):
//   exportSession(sessionId, { incomplete }) -> { ok, exported, exportName }
//   listSessions() -> [{ sessionId, meta }]
//   getSession(sessionId) -> { meta, trials }
//   markInterrupted(sessionId) -> { ok }
//
// Errors: every method rejects with StorageError (never resolves while claiming success when
// data was not persisted). `code` is one of STORAGE_ERROR_CODES.

export const STORAGE_ERROR_CODES = Object.freeze({
  QUOTA_EXCEEDED: "QUOTA_EXCEEDED",
  UNAVAILABLE: "UNAVAILABLE",
  SERIALIZATION: "SERIALIZATION",
  NOT_FOUND: "NOT_FOUND",
  INVALID: "INVALID",
});

export class StorageError extends Error {
  constructor(code, message, cause) {
    super(message);
    this.name = "StorageError";
    this.code = code;
    this.cause = cause;
  }
}

export const REQUIRED_METHODS = Object.freeze(["startSession", "saveTrial", "completeSession"]);

export function assertImplementsContract(adapter) {
  const missing = REQUIRED_METHODS.filter((name) => typeof adapter?.[name] !== "function");
  if (missing.length) throw new Error(`Storage adapter is missing required methods: ${missing.join(", ")}`);
  return adapter;
}

export function trialKey(trial) {
  return `${trial.sessionId}:${trial.trialNumber}`;
}
