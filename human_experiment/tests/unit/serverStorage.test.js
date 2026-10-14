// Server storage adapter (online collection) and conversion of the server files to CSV.
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { findConfigProblems } from "../../src/config/validateConfig.js";
import { createStorage } from "../../src/services/dataStorage/index.js";
import { parseCsv } from "../../src/services/dataStorage/csvSerializer.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { sessionCsvFromJsonl, sessionFromJsonl } from "../../src/services/dataStorage/serverRecords.js";
import { ServerStorage } from "../../src/services/dataStorage/serverStorage.js";
import { assertImplementsContract, StorageError } from "../../src/services/dataStorage/storageContract.js";
import { answer, makeEngine, runTraining, SCREENS, startToTraining } from "../helpers/engineHarness.js";

function jsonResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/** In-memory equivalent of public/save.php: one JSON-lines "file" per session. */
class FakeServer {
  constructor() {
    this.files = new Map();
    this.requests = [];
    this.failures = []; // queued failures: "network" or an HTTP status
  }

  fetch = async (url, init) => {
    const body = JSON.parse(init.body);
    this.requests.push(body);
    const failure = this.failures.shift();
    if (failure === "network") throw new TypeError("Failed to fetch");
    if (failure) return jsonResponse(failure, { ok: false, error: "fake failure" });
    return jsonResponse(...this.handle(body));
  };

  handle(body) {
    const line = (type, data, extra = {}) => `${JSON.stringify({ type, receivedAt: "2026-10-14T00:00:00Z", ...extra, data })}\n`;
    const id = body.session?.sessionId ?? body.trial?.sessionId ?? body.completion?.sessionId ?? body.sessionId;
    const file = this.files.get(id);
    if (body.action === "start") {
      if (file) return [200, { ok: true, duplicate: true }];
      this.files.set(id, line("start", body.session));
      return [200, { ok: true }];
    }
    if (file === undefined) return [404, { ok: false, error: "unknown session" }];
    if (body.action === "trial") {
      const key = `${id}:${body.trial.trialNumber}`;
      if (file.includes(`"trialKey":"${key}"`)) return [200, { ok: true, duplicate: true }];
      this.files.set(id, file + line("trial", body.trial, { trialKey: key }));
    } else if (body.action === "complete") this.files.set(id, file + line("complete", body.completion));
    else if (body.action === "interrupted") this.files.set(id, file + line("interrupted", { sessionId: id }));
    else return [400, { ok: false, error: "unknown action" }];
    return [200, { ok: true }];
  }
}

function serverAdapter(server = new FakeServer(), options = {}) {
  const sleeps = [];
  const storage = new ServerStorage({ endpoint: "save.php", fetch: server.fetch, sleep: async (ms) => sleeps.push(ms), ...options });
  return { storage, server, sleeps };
}

async function runSession(storage, driftTrials = 4) {
  const { engine, presenter } = makeEngine({ storage, overrides: { "drift.numberOfTrials": driftTrials } });
  await startToTraining(engine);
  await runTraining(engine, presenter, Array(20).fill(true));
  await engine.continueToDrift();
  const responses = ["A", "Probably A", "Probably B", "B"];
  let i = 0;
  while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, responses[i++ % 4]);
  return engine;
}

describe("server storage adapter", () => {
  it("implements the contract and is selected by storage.adapter = server", () => {
    const storage = createStorage({ storage: { adapter: "server", serverEndpoint: "save.php", localStoragePrefix: "hx" } }, { backup: null });
    expect(storage).toBeInstanceOf(ServerStorage);
    expect(storage.serverBacked).toBe(true);
    expect(() => assertImplementsContract(storage)).not.toThrow();
  });

  it("a full session reaches the server and converts to the same CSV as the pilot adapter", async () => {
    const { storage, server } = serverAdapter();
    const engine = await runSession(storage);
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE, savedToServer: true, exported: true });

    const memory = new MemoryStorage();
    await runSession(memory); // same seed and fake clock: identical session
    const file = server.files.get(engine.meta.sessionId);
    const converted = sessionCsvFromJsonl(file);
    expect(converted.warnings).toEqual([]);
    expect(converted.name).toBe(`${engine.meta.sessionId}.csv`);
    expect(converted.csv).toBe(memory.exports.at(-1).text);
    expect(parseCsv(converted.csv)).toHaveLength(engine.records.length);
  });

  it("retries network and server errors before succeeding", async () => {
    const { storage, server, sleeps } = serverAdapter();
    server.failures.push("network", 503);
    await expect(storage.startSession({ sessionId: "s1" })).resolves.toEqual({ ok: true });
    expect(server.requests).toHaveLength(3);
    expect(sleeps).toEqual([500, 1000]);
  });

  it("reports UNAVAILABLE after the last attempt; the engine's retry saves without duplicates", async () => {
    const { storage, server } = serverAdapter();
    const { engine, presenter } = makeEngine({ storage });
    await startToTraining(engine);
    server.failures.push("network", "network", "network");
    await answer(engine, presenter, "A");
    expect(engine.state).toMatchObject({ screen: SCREENS.ERROR, kind: "storage" });
    await engine.retry();
    await engine.retry(); // a second retry of the same record is acknowledged as a duplicate
    const lines = server.files.get(engine.meta.sessionId).trim().split("\n");
    expect(lines.filter((l) => JSON.parse(l).type === "trial")).toHaveLength(1);
    expect(engine.records).toHaveLength(1);
  });

  it("does not retry a request the server rejected", async () => {
    const { storage, server } = serverAdapter();
    server.failures.push(400);
    const error = await storage.startSession({ sessionId: "s1" }).catch((e) => e);
    expect(error).toBeInstanceOf(StorageError);
    expect(error.code).toBe("INVALID");
    expect(server.requests).toHaveLength(1);
    await expect(storage.saveTrial({ sessionId: "never-started", trialNumber: 1 })).rejects.toMatchObject({ code: "NOT_FOUND" });
  });

  it("a failing local backup never blocks the participant", async () => {
    const failing = new Proxy({}, { get: () => async () => { throw new Error("quota"); } });
    const { storage, server } = serverAdapter(undefined, { backup: failing });
    await storage.startSession({ sessionId: "s1" });
    await expect(storage.saveTrial({ sessionId: "s1", trialNumber: 1 })).resolves.toEqual({ ok: true });
    expect(storage.backupErrors.map((e) => e.method)).toEqual(["startSession", "saveTrial"]);
    expect(server.files.get("s1")).toContain('"trialKey":"s1:1"');
  });

  it("marking an interrupted session never fails, even offline", async () => {
    const { storage, server } = serverAdapter();
    await storage.startSession({ sessionId: "s1" });
    server.failures.push("network", "network", "network");
    await expect(storage.markInterrupted("s1")).resolves.toEqual({ ok: true });
    await expect(storage.markInterrupted("s1")).resolves.toEqual({ ok: true });
    expect(server.files.get("s1")).toContain('"type":"interrupted"');
  });
});

describe("server files to CSV", () => {
  const start = { type: "start", data: { sessionId: "s1", participantId: "P1", completionStatus: "in_progress" } };
  const trial = (n) => ({ type: "trial", trialKey: `s1:${n}`, data: { sessionId: "s1", trialNumber: n, phase: "training" } });
  const jsonl = (...records) => records.map((r) => (typeof r === "string" ? r : JSON.stringify(r))).join("\n");

  it("a session without a completion record is exported as interrupted and incomplete", () => {
    const session = sessionCsvFromJsonl(jsonl(start, trial(2), trial(1), trial(2)));
    expect(session.incomplete).toBe(true);
    expect(session.name).toBe("s1_INCOMPLETE.csv");
    expect(session.trials.map((t) => t.trialNumber)).toEqual([1, 2]);
    expect(parseCsv(session.csv)[0]).toMatchObject({ completionStatus: "interrupted", terminationReason: "no_completion_received", exportedAsIncomplete: "true" });
  });

  it("uses the completion record and skips malformed lines with a warning", () => {
    const session = sessionFromJsonl(jsonl(start, trial(1), "{broken", { type: "complete", data: { sessionId: "s1", completionStatus: "complete" } }));
    expect(session.incomplete).toBe(false);
    expect(session.meta.completionStatus).toBe("complete");
    expect(session.warnings).toEqual(["line 3: not valid JSON (skipped)"]);
  });

  it("a file without a start record yields no CSV", () => {
    expect(sessionCsvFromJsonl(jsonl(trial(1))).csv).toBeNull();
  });
});

describe("server configuration", () => {
  it("the shipped configuration is valid and uses the server adapter", () => {
    expect(findConfigProblems(experimentConfig)).toEqual([]);
    expect(experimentConfig.storage.adapter).toBe("server");
  });

  it("the Prolific completion link must be https", () => {
    const config = { ...experimentConfig, completion: { prolificCompletionUrl: "http://example.org" } };
    expect(findConfigProblems(config)).toContain("completion.prolificCompletionUrl must be empty or an https:// URL");
  });
});
