// Storage contract, localStorage adapter and CSV serialization (acceptance checks 10, 11, 13).
import { describe, expect, it } from "vitest";
import { createStorage } from "../../src/services/dataStorage/index.js";
import { CSV_COLUMNS, encodeField, parseCsv, serializeSessionCsv, stableStringify, TRIAL_COLUMNS } from "../../src/services/dataStorage/csvSerializer.js";
import { LocalCsvStorage } from "../../src/services/dataStorage/localCsvStorage.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { assertImplementsContract, StorageError } from "../../src/services/dataStorage/storageContract.js";
import { answer, correctResponse, makeEngine, runTraining, SCREENS, startToTraining } from "../helpers/engineHarness.js";

class FakeLocalStorage {
  constructor() {
    this.map = new Map();
    this.quotaBytes = Infinity;
    this.throwOnSet = null;
  }
  getItem(k) {
    return this.map.has(k) ? this.map.get(k) : null;
  }
  setItem(k, v) {
    if (this.throwOnSet) throw this.throwOnSet;
    const used = [...this.map.entries()].filter(([key]) => key !== k).reduce((n, [key, val]) => n + key.length + val.length, 0);
    if (used + k.length + v.length > this.quotaBytes) {
      const e = new Error("quota");
      e.name = "QuotaExceededError";
      throw e;
    }
    this.map.set(k, String(v));
  }
  removeItem(k) {
    this.map.delete(k);
  }
}

function localAdapter(ls = new FakeLocalStorage()) {
  const downloads = [];
  const storage = new LocalCsvStorage({ localStorage: ls, prefix: "hx", download: (name, text) => downloads.push({ name, text }) });
  return { storage, downloads, ls };
}

describe("CSV serialization (PRD §14)", () => {
  it("escapes commas, quotes and newlines (RFC 4180) and encodes booleans unambiguously", () => {
    expect(encodeField('a,b "c"\nd')).toBe('"a,b ""c""\nd"');
    expect(encodeField(true)).toBe("true");
    expect(encodeField(false)).toBe("false");
    expect(encodeField(null)).toBe("");
    expect(encodeField(0.1)).toBe("0.1");
  });
  it("serializes nested configuration deterministically", () => {
    expect(stableStringify({ b: 1, a: { d: [1, { z: 1, y: 2 }], c: null } })).toBe('{"a":{"c":null,"d":[1,{"y":2,"z":1}]},"b":1}');
  });
  it("contains every required trial field", () => {
    for (const field of ["participantId", "sessionId", "trialNumber", "phaseTrialNumber", "phase", "imageId", "trueGroup", "pc1", "pc2", "stimulusAngle", "currentAngleA", "currentAngleB", "response", "binaryResponse", "correct", "responseTimeMs", "feedbackShown", "timestamp"]) {
      expect(TRIAL_COLUMNS).toContain(field);
    }
  });
});

describe("full-session CSV (acceptance check 11)", () => {
  it("has one row per response, all columns, preserved responses and exportable metadata", async () => {
    const { engine, presenter, storage } = makeEngine({ overrides: { "drift.numberOfTrials": 6 } });
    await startToTraining(engine);
    const responses = ["A", "Probably A", "Probably B", "B"];
    await runTraining(engine, presenter, [false, false]);
    while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, correctResponse(engine));
    await engine.continueToDrift();
    let i = 0;
    while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, responses[i++ % 4]);
    expect(engine.state.screen).toBe(SCREENS.COMPLETE);
    const csv = storage.exports.at(-1);
    expect(csv.name).toBe(`${engine.meta.sessionId}.csv`);
    const rows = parseCsv(csv.text);
    expect(rows).toHaveLength(engine.records.length);
    expect(Object.keys(rows[0])).toEqual([...CSV_COLUMNS]);
    expect(rows.filter((r) => r.phase === "drift").map((r) => r.response)).toEqual(["A", "Probably A", "Probably B", "B", "A", "Probably A"]);
    const first = rows[0];
    expect(first).toMatchObject({
      completionStatus: "complete",
      consentGiven: "true",
      consentBypassed: "false",
      developmentMode: "false",
      sessionSeed: String(engine.meta.sessionSeed),
      trainingCriterionMet: "true",
      exportedAsIncomplete: "false",
      feedbackShown: "true",
    });
    expect(JSON.parse(first.effectiveConfigJson).drift.numberOfTrials).toBe(6);
    expect(JSON.parse(first.randomizationJson).driftSchedule).toHaveLength(6);
    expect(rows.every((r) => r.sessionEndTime === first.sessionEndTime && r.sessionEndTime !== "")).toBe(true);
    expect(rows.filter((r) => r.phase === "drift").every((r) => r.feedbackShown === "false")).toBe(true);
  });

  it("partial exports of an interrupted session are marked incomplete and derive counts from records", () => {
    const meta = { sessionId: "s1", completionStatus: "in_progress", trainingCriterionMet: false };
    const trials = [
      { sessionId: "s1", trialNumber: 2, phase: "training", criterionPassed: true, response: "B" },
      { sessionId: "s1", trialNumber: 1, phase: "training", criterionPassed: false, response: "A" },
      { sessionId: "s1", trialNumber: 3, phase: "drift", response: "A" },
    ];
    const rows = parseCsv(serializeSessionCsv(meta, trials, { incomplete: true }));
    expect(rows.map((r) => r.trialNumber)).toEqual(["1", "2", "3"]);
    expect(rows[0]).toMatchObject({ exportedAsIncomplete: "true", trainingCriterionMet: "true", trainingCriterionMetAtTrial: "2", trainingCompletedTrials: "2", driftCompletedTrials: "1" });
  });
});

describe("localStorage adapter (PRD §13, acceptance check 10)", () => {
  const meta = (id) => ({ sessionId: id, completionStatus: "in_progress", participantId: "p" });
  const trial = (id, n, phase = "training") => ({ sessionId: id, trialNumber: n, phase, response: "A" });

  it("implements the contract", () => {
    expect(() => assertImplementsContract(localAdapter().storage)).not.toThrow();
    expect(() => assertImplementsContract(new MemoryStorage())).not.toThrow();
    expect(() => assertImplementsContract({ startSession() {} })).toThrow();
  });

  it("saves to memory and localStorage, never downloads per trial, and ignores duplicate keys", async () => {
    const { storage, downloads, ls } = localAdapter();
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    const dup = await storage.saveTrial({ ...trial("s1", 1), response: "B" });
    expect(dup.duplicate).toBe(true);
    expect(JSON.parse(ls.getItem("hx:session:s1:trials"))).toHaveLength(1);
    expect(JSON.parse(ls.getItem("hx:session:s1:trials"))[0].response).toBe("A");
    expect(JSON.parse(ls.getItem("hx:session:s1:meta")).trainingCompletedTrials).toBe(1);
    expect(downloads).toHaveLength(0);
  });

  it("keeps sessions separate and never overwrites an existing session", async () => {
    const { storage, ls } = localAdapter();
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    await storage.startSession(meta("s2"));
    await expect(storage.startSession(meta("s1"))).rejects.toBeInstanceOf(StorageError);
    expect(JSON.parse(ls.getItem("hx:index"))).toEqual(["s1", "s2"]);
    expect(JSON.parse(ls.getItem("hx:session:s1:trials"))).toHaveLength(1);
  });

  it("recovers saved records after a refresh (new adapter instance) and exports them", async () => {
    const first = localAdapter();
    await first.storage.startSession(meta("s1"));
    await first.storage.saveTrial(trial("s1", 1));
    await first.storage.saveTrial(trial("s1", 2));
    const reloaded = localAdapter(first.ls);
    const sessions = await reloaded.storage.listSessions();
    expect(sessions.map((s) => [s.sessionId, s.meta.completionStatus])).toEqual([["s1", "in_progress"]]);
    await reloaded.storage.markInterrupted("s1");
    const result = await reloaded.storage.exportSession("s1", { incomplete: true });
    expect(result).toMatchObject({ exported: true, exportName: "s1_INCOMPLETE.csv" });
    const rows = parseCsv(reloaded.downloads[0].text);
    expect(rows).toHaveLength(2);
    expect(rows[0].completionStatus).toBe("interrupted");
    // Data stays available after export.
    expect(JSON.parse(first.ls.getItem("hx:session:s1:trials"))).toHaveLength(2);
  });

  it("surfaces quota errors and does not claim success", async () => {
    const { storage, ls } = localAdapter();
    await storage.startSession(meta("s1"));
    ls.quotaBytes = 10;
    await expect(storage.saveTrial(trial("s1", 1))).rejects.toMatchObject({ name: "StorageError", code: "QUOTA_EXCEEDED" });
    ls.quotaBytes = Infinity;
    expect((await storage.getSession("s1")).trials).toHaveLength(0);
    await storage.saveTrial(trial("s1", 1));
    expect((await storage.getSession("s1")).trials).toHaveLength(1);
  });

  it("reports unavailable storage", async () => {
    const { storage, ls } = localAdapter();
    ls.throwOnSet = new Error("SecurityError");
    await expect(storage.startSession(meta("s1"))).rejects.toMatchObject({ code: "UNAVAILABLE" });
  });

  it("a blocked download leaves data saved and reports exported:false", async () => {
    const ls = new FakeLocalStorage();
    const storage = new LocalCsvStorage({ localStorage: ls, download: () => { throw new Error("blocked"); } });
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    const result = await storage.completeSession({ sessionId: "s1", completionStatus: "complete" });
    expect(result).toMatchObject({ ok: true, exported: false, exportName: "s1.csv" });
    expect(JSON.parse(ls.getItem("hx:session:s1:meta")).completionStatus).toBe("complete");
  });
});

describe("replaceable adapter (acceptance check 13)", () => {
  it("the same engine runs unchanged on the localStorage adapter", async () => {
    const ls = new FakeLocalStorage();
    const downloads = [];
    const storage = createStorage({ storage: { adapter: "localCsv", localStoragePrefix: "hx" } }, { localStorage: ls, download: (n, t) => downloads.push({ n, t }) });
    const { engine, presenter } = makeEngine({ storage, overrides: { "drift.numberOfTrials": 4 } });
    await startToTraining(engine);
    await runTraining(engine, presenter, Array(20).fill(true));
    await engine.continueToDrift();
    while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, "B");
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE, exported: true });
    expect(downloads).toHaveLength(1);
    expect(parseCsv(downloads[0].t)).toHaveLength(24);
  });
});
