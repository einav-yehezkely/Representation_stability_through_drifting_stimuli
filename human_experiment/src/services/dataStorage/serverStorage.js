// Server-backed storage adapter for online collection (Prolific). Every record is POSTed to
// public/save.php, which appends it to one JSON-lines file per session outside the web root.
// The server is authoritative: a method resolves only after the server confirmed the write.
// A best-effort copy is also kept in this browser (LocalCsvStorage backup) for recovery and
// development; a failing backup never blocks the participant.
//
// Request body: { action: "start" | "trial" | "complete" | "interrupted", ... }
// Response:     { ok: true, duplicate?: boolean } or { ok: false, error: string }

import { STORAGE_ERROR_CODES, StorageError } from "./storageContract.js";

const defaultSleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class ServerStorage {
  /**
   * @param {object} options
   * @param {string} options.endpoint            URL of save.php (relative to the page).
   * @param {typeof fetch} [options.fetch]
   * @param {object} [options.backup]            Local adapter (LocalCsvStorage) or null.
   * @param {number} [options.timeoutMs]         Per-request timeout.
   * @param {number} [options.maxAttempts]       Attempts per call before the error is shown.
   * @param {(ms:number) => Promise<void>} [options.sleep]
   */
  constructor({ endpoint, fetch = globalThis.fetch?.bind(globalThis), backup = null, timeoutMs = 15000, maxAttempts = 3, sleep = defaultSleep }) {
    this.endpoint = endpoint;
    this.fetch = fetch;
    this.backup = backup;
    this.timeoutMs = timeoutMs;
    this.maxAttempts = maxAttempts;
    this.sleep = sleep;
    this.backupErrors = [];
    // Data reaches the researcher through the server; the UI shows no download for participants.
    this.serverBacked = true;
  }

  async post(body) {
    let lastError;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      if (attempt > 1) await this.sleep(500 * (attempt - 1));
      const controller = typeof AbortController === "function" ? new AbortController() : null;
      const timer = controller ? setTimeout(() => controller.abort(), this.timeoutMs) : null;
      let response;
      try {
        response = await this.fetch(this.endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          cache: "no-store",
          signal: controller?.signal,
        });
      } catch (error) {
        lastError = error; // network failure or timeout: retry
        continue;
      } finally {
        if (timer) clearTimeout(timer);
      }
      let reply = null;
      try {
        reply = await response.json();
      } catch {
        /* handled below */
      }
      if (response.ok && reply?.ok === true) return reply;
      const message = reply?.error ?? `HTTP ${response.status}`;
      // A rejected request will be rejected again; only server/transport errors are retried.
      if (response.status === 404) throw new StorageError(STORAGE_ERROR_CODES.NOT_FOUND, `The server did not find this session (${message}).`);
      if (response.status >= 400 && response.status < 500 && response.status !== 408 && response.status !== 429) {
        throw new StorageError(STORAGE_ERROR_CODES.INVALID, `The server rejected the data (${message}).`);
      }
      lastError = new Error(message);
    }
    throw new StorageError(STORAGE_ERROR_CODES.UNAVAILABLE, "The data could not be sent to the server.", lastError);
  }

  async onBackup(method, ...args) {
    if (!this.backup) return null;
    try {
      return await this.backup[method](...args);
    } catch (error) {
      this.backupErrors.push({ method, message: error.message });
      return null;
    }
  }

  async startSession(sessionMetadata) {
    if (!sessionMetadata?.sessionId) throw new StorageError(STORAGE_ERROR_CODES.INVALID, "Session metadata has no sessionId.");
    await this.post({ action: "start", session: sessionMetadata });
    await this.onBackup("startSession", sessionMetadata);
    return { ok: true };
  }

  async saveTrial(trial) {
    const reply = await this.post({ action: "trial", trial });
    await this.onBackup("saveTrial", trial);
    return reply.duplicate ? { ok: true, duplicate: true } : { ok: true };
  }

  async completeSession(completionMetadata) {
    await this.post({ action: "complete", completion: completionMetadata });
    await this.onBackup("updateSession", completionMetadata.sessionId, completionMetadata);
    return { ok: true, exported: true, savedToServer: true };
  }

  /** Recovery after a reload: mark the old session in this browser and, best effort, on the server. */
  async markInterrupted(sessionId) {
    await this.onBackup("markInterrupted", sessionId);
    try {
      await this.post({ action: "interrupted", sessionId });
    } catch {
      // The converter treats a session without a completion record as interrupted anyway.
    }
    return { ok: true };
  }

  async listSessions() {
    return (await this.onBackup("listSessions")) ?? [];
  }

  async getSession(sessionId) {
    if (!this.backup) throw new StorageError(STORAGE_ERROR_CODES.NOT_FOUND, "No local copy is kept.");
    return this.backup.getSession(sessionId);
  }

  /** Development only (partial export / download again): CSV of the local copy. */
  async exportSession(sessionId, options) {
    if (!this.backup) throw new StorageError(STORAGE_ERROR_CODES.NOT_FOUND, "No local copy is kept.");
    return this.backup.exportSession(sessionId, options);
  }
}
