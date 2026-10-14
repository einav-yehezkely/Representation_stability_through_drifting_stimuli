// Data sources of the admin page. Both return the same shapes:
//   listSessions() -> summary[]                (summary fields: see summarizeSession)
//   getSession(id) -> { meta, trials, status, lastActivity }
//   getSettings()  -> { values, updatedAt } | null      (experiment settings, see experimentSettings.js)
//   saveSettings(values) -> { values, updatedAt }
//
// ServerAdminSource reads the session files through public/admin_api.php (password login).
// LocalAdminSource reads the sessions saved in this browser's localStorage; it is used by the
// Vite dev server, which cannot run PHP.

import { loadExperimentSettings, saveLocalSettings } from "../config/experimentSettings.js";
import { LocalCsvStorage } from "../services/dataStorage/localCsvStorage.js";
import { sessionFromJsonl } from "../services/dataStorage/serverRecords.js";

/** A session without a completion record counts as in progress while it changed this recently (as in admin_api.php). */
export const ACTIVE_WINDOW_MS = 30 * 60 * 1000;

export class AdminAuthError extends Error {
  constructor(message = "Please sign in.") {
    super(message);
    this.name = "AdminAuthError";
  }
}

export class AdminRequestError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "AdminRequestError";
    this.status = status;
  }
}

/** An unfinished session that has been quiet for longer than ACTIVE_WINDOW_MS was abandoned. */
function liveStatus(storedStatus, lastActivity, now) {
  if (storedStatus !== "in_progress") return storedStatus ?? "interrupted";
  return lastActivity && now - Date.parse(lastActivity) < ACTIVE_WINDOW_MS ? "in_progress" : "interrupted";
}

/** Session summary computed from saved records; the same fields admin_api.php returns. */
export function summarizeSession({ sessionId, meta, trials, status, lastActivity }) {
  const training = trials.filter((t) => t.phase === "training");
  const drift = trials.filter((t) => t.phase === "drift");
  const lastWindow = training.findLast((t) => t.windowAccuracy !== null && t.windowAccuracy !== undefined);
  const rts = trials.map((t) => Number(t.responseTimeMs)).filter(Number.isFinite);
  return {
    sessionId,
    participantId: meta?.participantId ?? null,
    participantIdSource: meta?.participantIdSource ?? null,
    developmentMode: Boolean(meta?.developmentMode),
    sessionSeed: meta?.sessionSeed ?? null,
    sessionStartTime: meta?.sessionStartTime ?? null,
    sessionEndTime: meta?.sessionEndTime ?? null,
    status,
    terminationReason: meta?.terminationReason ?? null,
    trainingTrials: training.length,
    trainingCorrect: training.filter((t) => t.correct === true).length,
    trainingWindowAccuracy: lastWindow?.windowAccuracy ?? null,
    criterionMetAtTrial: training.find((t) => t.criterionPassed === true)?.trialNumber ?? null,
    driftTrials: drift.length,
    driftCorrect: drift.filter((t) => t.correct === true).length,
    driftTrialsPlanned: meta?.effectiveConfig?.drift?.numberOfTrials ?? null,
    meanResponseTimeMs: rts.length ? rts.reduce((a, b) => a + b, 0) / rts.length : null,
    lastTrialNumber: trials.reduce((max, t) => Math.max(max, t.trialNumber ?? 0), 0) || null,
    lastActivity,
  };
}

export class ServerAdminSource {
  constructor({ endpoint = "admin_api.php", fetch = globalThis.fetch?.bind(globalThis), now = () => Date.now() } = {}) {
    this.endpoint = endpoint;
    this.fetch = fetch;
    this.now = now;
    this.label = "Server data";
    this.requiresLogin = true;
  }

  async request(query, body) {
    const url = query ? `${this.endpoint}?${new URLSearchParams(query)}` : this.endpoint;
    let response;
    try {
      response = await this.fetch(url, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
        credentials: "same-origin",
        cache: "no-store",
      });
    } catch (error) {
      throw new AdminRequestError(`The server could not be reached (${error.message}).`, 0);
    }
    let reply = null;
    try {
      reply = await response.json();
    } catch {
      /* handled below */
    }
    if (response.status === 401 && !body) throw new AdminAuthError();
    if (!response.ok || reply?.ok !== true) throw new AdminRequestError(reply?.error ?? `HTTP ${response.status}`, response.status);
    return reply;
  }

  async status() {
    return this.request({ action: "status" });
  }

  async login(password) {
    return this.request(null, { action: "login", password });
  }

  async logout() {
    return this.request(null, { action: "logout" });
  }

  async listSessions() {
    return (await this.request({ action: "sessions" })).sessions;
  }

  async getSettings() {
    return (await this.request({ action: "settings" })).settings;
  }

  async saveSettings(values) {
    return (await this.request(null, { action: "save_settings", values })).settings;
  }

  async getSession(sessionId) {
    const reply = await this.request({ action: "session", id: sessionId });
    const parsed = sessionFromJsonl(reply.jsonl);
    if (!parsed.meta) throw new AdminRequestError("The session file has no start record.", 500);
    const meta = { ...parsed.meta };
    // sessionFromJsonl marks every session without a completion record as interrupted; one that
    // is still being written to is in progress.
    const stored = meta.terminationReason === "no_completion_received" ? "in_progress" : meta.completionStatus;
    const status = liveStatus(stored, reply.lastActivity, this.now());
    if (status === "in_progress") Object.assign(meta, { completionStatus: "in_progress", terminationReason: null });
    return { meta, trials: parsed.trials, status, lastActivity: reply.lastActivity, warnings: parsed.warnings };
  }
}

export class LocalAdminSource {
  constructor({ prefix = "hx", localStorage = globalThis.localStorage, now = () => Date.now() } = {}) {
    this.prefix = prefix;
    this.localStorage = localStorage;
    this.storage = new LocalCsvStorage({ prefix, localStorage, download: () => {} });
    this.now = now;
    this.label = "This browser (development)";
    this.requiresLogin = false;
  }

  async getSettings() {
    const { values, updatedAt } = await loadExperimentSettings({ source: "local", localStorage: this.localStorage, prefix: this.prefix });
    return values ? { values, updatedAt } : null;
  }

  async saveSettings(values) {
    return saveLocalSettings(values, { localStorage: this.localStorage, prefix: this.prefix });
  }

  async getSession(sessionId) {
    this.storage.refresh(sessionId); // always read the latest saved state
    const { meta, trials } = await this.storage.getSession(sessionId);
    const lastActivity = meta.lastSavedAt ?? meta.sessionStartTime ?? null;
    return { meta, trials, status: liveStatus(meta.completionStatus, lastActivity, this.now()), lastActivity, warnings: [] };
  }

  async listSessions() {
    const sessions = [];
    for (const { sessionId } of await this.storage.listSessions()) {
      try {
        sessions.push(summarizeSession({ sessionId, ...(await this.getSession(sessionId)) }));
      } catch {
        // unreadable entry: skipped
      }
    }
    return sessions.sort((a, b) => String(b.sessionStartTime).localeCompare(String(a.sessionStartTime)));
  }
}
