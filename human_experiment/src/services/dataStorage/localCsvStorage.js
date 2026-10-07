// Development/pilot storage adapter: in-memory + localStorage, CSV download at completion.
//
// THIS IS NOT THE FINAL PROLIFIC DATA COLLECTION MECHANISM. A browser download stays on the
// participant's computer; final online collection needs a server-backed adapter.
//
// localStorage layout (prefix defaults to "hx"):
//   <prefix>:index                 JSON array of sessionIds (append-only)
//   <prefix>:session:<id>:meta     session metadata
//   <prefix>:session:<id>:trials   JSON array of trial records

import { csvFileName, serializeSessionCsv } from "./csvSerializer.js";
import { STORAGE_ERROR_CODES, StorageError, trialKey } from "./storageContract.js";

function isQuotaError(error) {
  return (
    error &&
    (error.name === "QuotaExceededError" || error.name === "NS_ERROR_DOM_QUOTA_REACHED" || error.code === 22 || error.code === 1014)
  );
}

export class LocalCsvStorage {
  /**
   * @param {object} options
   * @param {Storage} [options.localStorage]
   * @param {string} [options.prefix]
   * @param {(name:string, text:string) => void} options.download
   */
  constructor({ localStorage = globalThis.localStorage, prefix = "hx", download }) {
    this.ls = localStorage;
    this.prefix = prefix;
    this.download = download;
    this.sessions = new Map(); // sessionId -> { meta, trials, keys:Set }
  }

  key(...parts) {
    return [this.prefix, ...parts].join(":");
  }

  write(key, value) {
    let text;
    try {
      text = JSON.stringify(value);
    } catch (error) {
      throw new StorageError(STORAGE_ERROR_CODES.SERIALIZATION, "Data could not be serialized for storage.", error);
    }
    try {
      if (!this.ls) throw new Error("localStorage is not available");
      this.ls.setItem(key, text);
    } catch (error) {
      if (isQuotaError(error)) {
        throw new StorageError(STORAGE_ERROR_CODES.QUOTA_EXCEEDED, "Browser storage is full; the data could not be saved.", error);
      }
      throw new StorageError(STORAGE_ERROR_CODES.UNAVAILABLE, "Browser storage is unavailable; the data could not be saved.", error);
    }
  }

  read(key) {
    let text;
    try {
      if (!this.ls) throw new Error("localStorage is not available");
      text = this.ls.getItem(key);
    } catch (error) {
      throw new StorageError(STORAGE_ERROR_CODES.UNAVAILABLE, "Browser storage is unavailable.", error);
    }
    if (text === null) return null;
    try {
      return JSON.parse(text);
    } catch (error) {
      throw new StorageError(STORAGE_ERROR_CODES.SERIALIZATION, `Stored data under ${key} is corrupted.`, error);
    }
  }

  readIndex() {
    return this.read(this.key("index")) ?? [];
  }

  /** Load a session into memory from localStorage (e.g. after a refresh). */
  load(sessionId) {
    if (this.sessions.has(sessionId)) return this.sessions.get(sessionId);
    const meta = this.read(this.key("session", sessionId, "meta"));
    if (!meta) throw new StorageError(STORAGE_ERROR_CODES.NOT_FOUND, `Session ${sessionId} was not found.`);
    const trials = this.read(this.key("session", sessionId, "trials")) ?? [];
    const entry = { meta, trials, keys: new Set(trials.map(trialKey)) };
    this.sessions.set(sessionId, entry);
    return entry;
  }

  async startSession(sessionMetadata) {
    const { sessionId } = sessionMetadata ?? {};
    if (!sessionId) throw new StorageError(STORAGE_ERROR_CODES.INVALID, "Session metadata has no sessionId.");
    const index = this.readIndex();
    if (index.includes(sessionId) || this.sessions.has(sessionId)) {
      throw new StorageError(STORAGE_ERROR_CODES.INVALID, `Session ${sessionId} already exists; it will not be overwritten.`);
    }
    const meta = { ...sessionMetadata, lastSavedAt: new Date().toISOString() };
    this.write(this.key("session", sessionId, "meta"), meta);
    this.write(this.key("session", sessionId, "trials"), []);
    this.write(this.key("index"), [...index, sessionId]);
    this.sessions.set(sessionId, { meta, trials: [], keys: new Set() });
    return { ok: true };
  }

  async saveTrial(trial) {
    const entry = this.load(trial.sessionId);
    const key = trialKey(trial);
    if (entry.keys.has(key)) return { ok: true, duplicate: true };
    const trials = [...entry.trials, Object.freeze({ ...trial })];
    const meta = {
      ...entry.meta,
      lastTrialNumber: trial.trialNumber,
      currentPhase: trial.phase,
      trainingCompletedTrials: trials.filter((t) => t.phase === "training").length,
      driftCompletedTrials: trials.filter((t) => t.phase === "drift").length,
      lastSavedAt: new Date().toISOString(),
    };
    const previousTrials = entry.trials;
    this.write(this.key("session", trial.sessionId, "trials"), trials);
    try {
      this.write(this.key("session", trial.sessionId, "meta"), meta);
    } catch (error) {
      // Roll back the trial list so storage stays consistent with what was reported.
      try {
        this.write(this.key("session", trial.sessionId, "trials"), previousTrials);
      } catch {
        /* the original error is reported below */
      }
      throw error;
    }
    entry.trials = trials;
    entry.meta = meta;
    entry.keys.add(key);
    return { ok: true };
  }

  async updateSession(sessionId, patch) {
    const entry = this.load(sessionId);
    const meta = { ...entry.meta, ...patch, lastSavedAt: new Date().toISOString() };
    this.write(this.key("session", sessionId, "meta"), meta);
    entry.meta = meta;
    return { ok: true };
  }

  async completeSession(completionMetadata) {
    const { sessionId } = completionMetadata;
    await this.updateSession(sessionId, completionMetadata);
    const incomplete = completionMetadata.completionStatus !== "complete";
    return this.exportSession(sessionId, { incomplete });
  }

  async exportSession(sessionId, { incomplete } = {}) {
    const entry = this.load(sessionId);
    const markIncomplete = incomplete ?? entry.meta.completionStatus !== "complete";
    const name = csvFileName(sessionId, { incomplete: markIncomplete });
    const text = serializeSessionCsv(entry.meta, entry.trials, { incomplete: markIncomplete });
    try {
      this.download(name, text);
    } catch (error) {
      return { ok: true, exported: false, exportName: name, exportError: error.message };
    }
    return { ok: true, exported: true, exportName: name };
  }

  async markInterrupted(sessionId) {
    const entry = this.load(sessionId);
    if (entry.meta.completionStatus !== "in_progress") return { ok: true };
    return this.updateSession(sessionId, {
      completionStatus: "interrupted",
      terminationReason: "page_reloaded_or_closed_before_completion",
    });
  }

  async listSessions() {
    return this.readIndex().map((sessionId) => {
      try {
        return { sessionId, meta: this.load(sessionId).meta };
      } catch (error) {
        return { sessionId, meta: null, error: error.message };
      }
    });
  }

  async getSession(sessionId) {
    const entry = this.load(sessionId);
    return { meta: entry.meta, trials: entry.trials };
  }

  /** Forget the in-memory copy so the next read comes from localStorage (live review). */
  refresh(sessionId) {
    this.sessions.delete(sessionId);
  }
}
