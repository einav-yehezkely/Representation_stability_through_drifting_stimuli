// Admin data sources: summaries, live status and the server client.
import { describe, expect, it } from "vitest";
import { AdminAuthError, AdminRequestError, LocalAdminSource, ServerAdminSource, summarizeSession } from "../../src/admin/dataSource.js";
import { rollingAccuracy } from "../../src/admin/accuracyChart.js";

class MapStorage {
  constructor() {
    this.map = new Map();
  }
  getItem(k) {
    return this.map.has(k) ? this.map.get(k) : null;
  }
  setItem(k, v) {
    this.map.set(k, String(v));
  }
}

const NOW = Date.parse("2026-10-14T12:00:00Z");
const trial = (n, phase, correct, extra = {}) => ({ sessionId: "s1", trialNumber: n, phaseTrialNumber: n, phase, correct, responseTimeMs: 400 + n, windowAccuracy: null, criterionPassed: null, ...extra });

describe("summarizeSession", () => {
  it("counts phases, correctness, criterion and response times", () => {
    const trials = [trial(1, "training", true), trial(2, "training", false, { windowAccuracy: 0.5, criterionPassed: false }), trial(3, "training", true, { windowAccuracy: 1, criterionPassed: true }), trial(4, "drift", true), trial(5, "drift", false)];
    const s = summarizeSession({ sessionId: "s1", meta: { participantId: "P", developmentMode: false, effectiveConfig: { drift: { numberOfTrials: 360 } } }, trials, status: "in_progress", lastActivity: null });
    expect(s).toMatchObject({ participantId: "P", trainingTrials: 3, trainingCorrect: 2, trainingWindowAccuracy: 1, criterionMetAtTrial: 3, driftTrials: 2, driftCorrect: 1, driftTrialsPlanned: 360, lastTrialNumber: 5 });
    expect(s.meanResponseTimeMs).toBe(403);
  });
});

describe("LocalAdminSource", () => {
  it("lists localStorage sessions; an unfinished session goes from in progress to interrupted after 30 quiet minutes", async () => {
    const ls = new MapStorage();
    const meta = (id, lastSavedAt, completionStatus = "in_progress") => ({ sessionId: id, sessionStartTime: lastSavedAt, lastSavedAt, completionStatus, participantId: id });
    ls.setItem("hx:index", JSON.stringify(["recent", "old", "done"]));
    ls.setItem("hx:session:recent:meta", JSON.stringify(meta("recent", "2026-10-14T11:50:00Z")));
    ls.setItem("hx:session:old:meta", JSON.stringify(meta("old", "2026-10-14T10:00:00Z")));
    ls.setItem("hx:session:done:meta", JSON.stringify(meta("done", "2026-10-14T09:00:00Z", "complete")));
    for (const id of ["recent", "old", "done"]) ls.setItem(`hx:session:${id}:trials`, "[]");
    const source = new LocalAdminSource({ localStorage: ls, now: () => NOW });
    const list = await source.listSessions();
    expect(list.map((s) => [s.sessionId, s.status])).toEqual([
      ["recent", "in_progress"],
      ["old", "interrupted"],
      ["done", "complete"],
    ]);
  });
});

describe("ServerAdminSource", () => {
  const fakeFetch = (routes, log = []) => async (url, init) => {
    log.push({ url, init });
    const key = init.method === "POST" ? `POST ${JSON.parse(init.body).action}` : url.split("?")[1];
    const [status, body] = routes[key] ?? [404, { ok: false, error: "no route" }];
    return { ok: status < 400, status, json: async () => body };
  };

  it("logs in with a JSON POST and maps 401 on reads to AdminAuthError", async () => {
    const log = [];
    const source = new ServerAdminSource({
      fetch: fakeFetch({ "POST login": [401, { ok: false, error: "Incorrect password." }], "action=sessions": [401, { ok: false, error: "login required" }] }, log),
    });
    await expect(source.login("x")).rejects.toThrow(new AdminRequestError("Incorrect password.", 401));
    expect(log[0].init.headers["Content-Type"]).toBe("application/json");
    await expect(source.listSessions()).rejects.toBeInstanceOf(AdminAuthError);
  });

  it("parses a session file; one without a completion record is in progress while recently written", async () => {
    const id = "11111111-2222-4333-8444-555555555555";
    const jsonl = [
      { type: "start", data: { sessionId: id, participantId: "P", completionStatus: "in_progress", effectiveConfig: { drift: { numberOfTrials: 4 } } } },
      { type: "trial", trialKey: `${id}:2`, data: { ...trial(2, "training", false), sessionId: id } },
      { type: "trial", trialKey: `${id}:1`, data: { ...trial(1, "training", true), sessionId: id } },
    ]
      .map((l) => JSON.stringify(l))
      .join("\n");
    const reply = (lastActivity) => [200, { ok: true, sessionId: id, jsonl, lastActivity }];
    const recent = new ServerAdminSource({ now: () => NOW, fetch: fakeFetch({ [`action=session&id=${id}`]: reply("2026-10-14T11:59:00Z") }) });
    const session = await recent.getSession(id);
    expect(session.status).toBe("in_progress");
    expect(session.meta.completionStatus).toBe("in_progress");
    expect(session.trials.map((t) => t.trialNumber)).toEqual([1, 2]);
    const stale = new ServerAdminSource({ now: () => NOW, fetch: fakeFetch({ [`action=session&id=${id}`]: reply("2026-10-14T09:00:00Z") }) });
    expect((await stale.getSession(id)).status).toBe("interrupted");
  });
});

describe("rollingAccuracy", () => {
  it("uses the last `window` answers (fewer at the start)", () => {
    const records = [true, false, true, true, false].map((correct) => ({ correct }));
    expect(rollingAccuracy(records, 3).map((p) => [p.accuracy, p.n])).toEqual([
      [1, 1],
      [0.5, 2],
      [2 / 3, 3],
      [2 / 3, 3],
      [2 / 3, 3],
    ]);
  });
});
