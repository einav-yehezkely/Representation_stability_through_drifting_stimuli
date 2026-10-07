// In-memory storage adapter implementing the same contract. Used by tests and as a
// demonstration that adapters are replaceable without touching phase or scientific logic.

import { csvFileName, serializeSessionCsv } from "./csvSerializer.js";
import { STORAGE_ERROR_CODES, StorageError, trialKey } from "./storageContract.js";

export class MemoryStorage {
  constructor({ onExport } = {}) {
    this.sessions = new Map();
    this.exports = [];
    this.onExport = onExport;
  }

  async startSession(sessionMetadata) {
    if (this.sessions.has(sessionMetadata.sessionId)) {
      throw new StorageError(STORAGE_ERROR_CODES.INVALID, "Session already exists.");
    }
    this.sessions.set(sessionMetadata.sessionId, { meta: { ...sessionMetadata }, trials: [], keys: new Set() });
    return { ok: true };
  }

  entry(sessionId) {
    const entry = this.sessions.get(sessionId);
    if (!entry) throw new StorageError(STORAGE_ERROR_CODES.NOT_FOUND, "Unknown session.");
    return entry;
  }

  async saveTrial(trial) {
    const entry = this.entry(trial.sessionId);
    const key = trialKey(trial);
    if (entry.keys.has(key)) return { ok: true, duplicate: true };
    entry.trials.push(Object.freeze({ ...trial }));
    entry.keys.add(key);
    return { ok: true };
  }

  async completeSession(completionMetadata) {
    const entry = this.entry(completionMetadata.sessionId);
    Object.assign(entry.meta, completionMetadata);
    return this.exportSession(completionMetadata.sessionId, { incomplete: completionMetadata.completionStatus !== "complete" });
  }

  async exportSession(sessionId, { incomplete } = {}) {
    const entry = this.entry(sessionId);
    const name = csvFileName(sessionId, { incomplete });
    const text = serializeSessionCsv(entry.meta, entry.trials, { incomplete });
    this.exports.push({ name, text });
    this.onExport?.(name, text);
    return { ok: true, exported: true, exportName: name };
  }

  async getSession(sessionId) {
    const entry = this.entry(sessionId);
    return { meta: entry.meta, trials: entry.trials };
  }

  async listSessions() {
    return [...this.sessions.entries()].map(([sessionId, entry]) => ({ sessionId, meta: entry.meta }));
  }

  async markInterrupted(sessionId) {
    const entry = this.entry(sessionId);
    if (entry.meta.completionStatus === "in_progress") entry.meta.completionStatus = "interrupted";
    return { ok: true };
  }
}
