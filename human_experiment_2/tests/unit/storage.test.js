// Storage contract, localStorage adapter and CSV serialization.
import { describe, expect, it } from "vitest";
import { createStorage } from "../../src/services/dataStorage/index.js";
import { CSV_COLUMNS, encodeField, parseCsv, serializeSessionCsv, stableStringify, TRIAL_COLUMNS } from "../../src/services/dataStorage/csvSerializer.js";
import { LocalCsvStorage } from "../../src/services/dataStorage/localCsvStorage.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { assertImplementsContract, StorageError } from "../../src/services/dataStorage/storageContract.js";
import { makeEngine, runAll, SCREENS, startExperiment } from "../helpers/engineHarness.js";

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
  const storage = new LocalCsvStorage({ localStorage: ls, prefix: "hx2", download: (name, text) => downloads.push({ name, text }) });
  return { storage, downloads, ls };
}

describe("CSV serialization", () => {
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
    for (const field of ["participantId", "sessionId", "trialNumber", "sampleGroup", "sampleSourceTrial", "sampleImageId", "optionAImageId", "optionBImageId", "leftGroup", "currentAngleA", "currentAngleB", "response", "chosenGroup", "chosenImageId", "correct", "responseTimeMs", "sampleDurationMs", "blankDurationMs", "interTrialDurationMs", "timestamp"]) {
      expect(TRIAL_COLUMNS).toContain(field);
    }
  });
});

describe("full-session CSV", () => {
  it("has one row per response, all columns, preserved responses and exportable metadata", async () => {
    const { engine, presenter, storage } = makeEngine({ overrides: { "drift.numberOfTrials": 6 } });
    await startExperiment(engine);
    const responses = ["left", "right", "right", "left", "left", "right"];
    await runAll(engine, presenter, (_, i) => responses[i]);
    expect(engine.state.screen).toBe(SCREENS.COMPLETE);
    const csv = storage.exports.at(-1);
    expect(csv.name).toBe(`${engine.meta.sessionId}.csv`);
    const rows = parseCsv(csv.text);
    expect(rows).toHaveLength(engine.records.length);
    expect(Object.keys(rows[0])).toEqual([...CSV_COLUMNS]);
    expect(rows.map((r) => r.response)).toEqual(responses);
    const first = rows[0];
    expect(first).toMatchObject({
      completionStatus: "complete",
      consentGiven: "true",
      consentBypassed: "false",
      developmentMode: "false",
      sessionSeed: String(engine.meta.sessionSeed),
      completedTrials: "6",
      exportedAsIncomplete: "false",
    });
    expect(JSON.parse(first.effectiveConfigJson).drift.numberOfTrials).toBe(6);
    expect(JSON.parse(first.randomizationJson).sampleSchedule).toHaveLength(6);
    expect(rows.every((r) => r.sessionEndTime === first.sessionEndTime && r.sessionEndTime !== "")).toBe(true);
  });

  it("partial exports of an interrupted session are marked incomplete and derive counts from records", () => {
    const meta = { sessionId: "s1", completionStatus: "in_progress" };
    const trials = [
      { sessionId: "s1", trialNumber: 2, response: "right" },
      { sessionId: "s1", trialNumber: 1, response: "left" },
      { sessionId: "s1", trialNumber: 3, response: "left" },
    ];
    const rows = parseCsv(serializeSessionCsv(meta, trials, { incomplete: true }));
    expect(rows.map((r) => r.trialNumber)).toEqual(["1", "2", "3"]);
    expect(rows[0]).toMatchObject({ exportedAsIncomplete: "true", completedTrials: "3" });
  });
});

describe("localStorage adapter", () => {
  const meta = (id) => ({ sessionId: id, completionStatus: "in_progress", participantId: "p" });
  const trial = (id, n) => ({ sessionId: id, trialNumber: n, response: "left" });

  it("implements the contract", () => {
    expect(() => assertImplementsContract(localAdapter().storage)).not.toThrow();
    expect(() => assertImplementsContract(new MemoryStorage())).not.toThrow();
    expect(() => assertImplementsContract({ startSession() {} })).toThrow();
  });

  it("saves to memory and localStorage, never downloads per trial, and ignores duplicate keys", async () => {
    const { storage, downloads, ls } = localAdapter();
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    const dup = await storage.saveTrial({ ...trial("s1", 1), response: "right" });
    expect(dup.duplicate).toBe(true);
    expect(JSON.parse(ls.getItem("hx2:session:s1:trials"))).toHaveLength(1);
    expect(JSON.parse(ls.getItem("hx2:session:s1:trials"))[0].response).toBe("left");
    expect(JSON.parse(ls.getItem("hx2:session:s1:meta")).completedTrials).toBe(1);
    expect(downloads).toHaveLength(0);
  });

  it("keeps sessions separate and never overwrites an existing session", async () => {
    const { storage, ls } = localAdapter();
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    await storage.startSession(meta("s2"));
    await expect(storage.startSession(meta("s1"))).rejects.toBeInstanceOf(StorageError);
    expect(JSON.parse(ls.getItem("hx2:index"))).toEqual(["s1", "s2"]);
    expect(JSON.parse(ls.getItem("hx2:session:s1:trials"))).toHaveLength(1);
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
    expect(JSON.parse(first.ls.getItem("hx2:session:s1:trials"))).toHaveLength(2);
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

  it("completion saves the metadata and downloads nothing; a blocked export on request reports exported:false", async () => {
    const ls = new FakeLocalStorage();
    const downloads = [];
    const storage = new LocalCsvStorage({ localStorage: ls, download: (name) => downloads.push(name) });
    await storage.startSession(meta("s1"));
    await storage.saveTrial(trial("s1", 1));
    expect(await storage.completeSession({ sessionId: "s1", completionStatus: "complete" })).toEqual({ ok: true });
    expect(downloads).toEqual([]);
    expect(JSON.parse(ls.getItem("hx2:session:s1:meta")).completionStatus).toBe("complete");
    storage.download = () => {
      throw new Error("blocked");
    };
    expect(await storage.exportSession("s1")).toMatchObject({ ok: true, exported: false, exportName: "s1.csv" });
  });
});

describe("replaceable adapter", () => {
  it("the same engine runs unchanged on the localStorage adapter", async () => {
    const ls = new FakeLocalStorage();
    const downloads = [];
    const storage = createStorage({ storage: { adapter: "localCsv", localStoragePrefix: "hx2" } }, { localStorage: ls, download: (n, t) => downloads.push({ n, t }) });
    const { engine, presenter } = makeEngine({ storage, overrides: { "drift.numberOfTrials": 4 } });
    await startExperiment(engine);
    await runAll(engine, presenter, () => "right");
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE });
    expect(downloads).toHaveLength(0); // nothing is downloaded at completion
    await storage.exportSession(engine.meta.sessionId);
    expect(parseCsv(downloads[0].t)).toHaveLength(4);
  });
});
