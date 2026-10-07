// Engine integration tests with fakes (acceptance checks 4-10, 13).
import { describe, expect, it } from "vitest";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { StorageError } from "../../src/services/dataStorage/storageContract.js";
import {
  answer,
  correctResponse,
  makeEngine,
  runTraining,
  SCREENS,
  startToTraining,
  wrongResponse,
} from "../helpers/engineHarness.js";

const allCorrect = (n) => Array(n).fill(true);

async function passTraining(engine, presenter) {
  await runTraining(engine, presenter, allCorrect(20));
  expect(engine.state.screen).toBe(SCREENS.TRANSITION);
}

async function runDrift(engine, presenter, responder = correctResponse) {
  await engine.continueToDrift();
  while (engine.state.screen === SCREENS.TRIAL) {
    await answer(engine, presenter, responder(engine));
  }
}

describe("flow", () => {
  it("consent -> instructions -> training", async () => {
    const { engine } = makeEngine();
    expect(engine.state.screen).toBe(SCREENS.CONSENT);
    await engine.acceptConsent();
    expect(engine.state.screen).toBe(SCREENS.INSTRUCTIONS);
    expect(engine.meta.consentGiven).toBe(true);
    expect(engine.meta.consentTimestamp).toMatch(/Z$/);
    await engine.startTraining();
    expect(engine.state).toMatchObject({ screen: SCREENS.TRIAL, status: "responding", phase: "training" });
  });

  it("consent bypass is only possible in development mode and is recorded as a bypass", async () => {
    const participant = makeEngine();
    await expect(participant.engine.bypassConsent()).rejects.toThrow();
    const dev = makeEngine({ developmentMode: true });
    await dev.engine.bypassConsent();
    expect(dev.engine.meta).toMatchObject({ consentGiven: false, consentBypassed: true, developmentMode: true });
    const stored = await dev.storage.getSession(dev.engine.meta.sessionId);
    expect(stored.meta.consentBypassed).toBe(true);
    expect(stored.meta.consentGiven).toBe(false);
  });
});

describe("training (acceptance checks 4, 5)", () => {
  it("cannot pass before 20 responses even when all are correct", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    await runTraining(engine, presenter, allCorrect(19));
    expect(engine.state.screen).toBe(SCREENS.TRIAL);
    expect(engine.criterion.passed).toBe(false);
    await answer(engine, presenter, correctResponse(engine));
    expect(engine.state.screen).toBe(SCREENS.TRANSITION);
    expect(engine.trainingRecords).toHaveLength(20);
  });

  it("16/20 passes and 15/20 fails", async () => {
    const pass = makeEngine();
    await startToTraining(pass.engine);
    await runTraining(pass.engine, pass.presenter, [...Array(4).fill(false), ...allCorrect(16)]);
    expect(pass.engine.state.screen).toBe(SCREENS.TRANSITION);

    const fail = makeEngine();
    await startToTraining(fail.engine);
    await runTraining(fail.engine, fail.presenter, [...Array(5).fill(false), ...allCorrect(15)]);
    expect(fail.engine.state.screen).toBe(SCREENS.TRIAL);
    expect(fail.engine.trainingRecords.at(-1).windowAccuracy).toBe(0.75);
  });

  it("high cumulative accuracy with a failing last-20 window does not pass", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "training.minTrials": 40 } });
    await startToTraining(engine);
    // 25 correct, then a last window of 15/20 (40 trials, cumulative 77.5%... last window 75%).
    const pattern = [...allCorrect(20), ...Array(5).fill(false), ...allCorrect(15)];
    await runTraining(engine, presenter, pattern);
    expect(engine.trainingRecords).toHaveLength(40);
    expect(engine.trainingRecords.filter((r) => r.correct).length / 40).toBeGreaterThanOrEqual(0.8);
    expect(engine.state.screen).toBe(SCREENS.TRIAL);
    expect(engine.criterion.passed).toBe(false);
  });

  it("keeps centers fixed, shows feedback on every training trial including the qualifying one", async () => {
    const { engine, presenter, clock, config } = makeEngine();
    await startToTraining(engine);
    const states = [];
    engine.subscribe((s) => states.push(s));
    await passTraining(engine, presenter);
    const records = engine.trainingRecords;
    for (const r of records) {
      expect(r.feedbackShown).toBe(true);
      expect([r.currentAngleA, r.currentAngleB]).toEqual([0, 180]);
      expect([r.centerA_pc1, r.centerA_pc2]).toEqual(engine.initial.centerA);
      expect([r.centerB_pc1, r.centerB_pc2]).toEqual(engine.initial.centerB);
    }
    const feedbackStates = states.filter((s) => s.status === "feedback");
    expect(feedbackStates).toHaveLength(20);
    expect(feedbackStates.at(-1).feedback).toBe("Correct");
    expect(clock.sleeps).toEqual(Array(20).fill(config.ui.feedbackDurationMs));
    // The transition comes after the qualifying trial's feedback.
    const lastFeedback = states.lastIndexOf(feedbackStates.at(-1));
    expect(states[lastFeedback + 1].screen).toBe(SCREENS.TRANSITION);
  });

  it("drift waits for Continue", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    await passTraining(engine, presenter);
    const shownBefore = presenter.shown.length;
    await answer(engine, presenter, "A"); // ignored: no trial is active
    expect(engine.records).toHaveLength(20);
    expect(presenter.shown.length).toBe(shownBefore);
    await engine.continueToDrift();
    expect(engine.state).toMatchObject({ screen: SCREENS.TRIAL, phase: "drift" });
  });

  it("incorrect feedback text is shown for wrong answers", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    const states = [];
    engine.subscribe((s) => states.push(s));
    await answer(engine, presenter, wrongResponse(engine));
    expect(states.find((s) => s.status === "feedback").feedback).toBe("Incorrect");
  });
});

describe("drift (acceptance checks 6, 7)", () => {
  it("starts at the initial centers, moves after each response, ends after exactly 360 trials", async () => {
    const { engine, presenter, storage } = makeEngine();
    await startToTraining(engine);
    await passTraining(engine, presenter);
    const states = [];
    engine.subscribe((s) => states.push(s));
    await runDrift(engine, presenter);
    const drift = engine.driftRecords;
    expect(drift).toHaveLength(360);
    const byTrial = (k) => drift[k - 1];
    expect([byTrial(1).currentAngleA, byTrial(1).currentAngleB]).toEqual([0, 180]);
    expect([byTrial(2).currentAngleA, byTrial(2).currentAngleB]).toEqual([1, 181]);
    expect([byTrial(180).currentAngleA, byTrial(180).currentAngleB]).toEqual([179, 359]);
    expect([byTrial(181).currentAngleA, byTrial(181).currentAngleB]).toEqual([180, 0]);
    expect([byTrial(360).currentAngleA, byTrial(360).currentAngleB]).toEqual([359, 179]);
    expect(drift.map((r) => r.currentAngleA)).toEqual([...Array(360).keys()]);
    // First drift trial uses the training centers exactly.
    expect([byTrial(1).centerA_pc1, byTrial(1).centerA_pc2]).toEqual(engine.initial.centerA);
    // Actual center angle tracks the nominal angle with the source offset (center at 359.49°).
    const offset = byTrial(1).centerA_actualAngle - 360;
    for (const r of drift) {
      const diff = (((r.centerA_actualAngle - r.currentAngleA - offset) % 360) + 540) % 360 - 180;
      expect(Math.abs(diff)).toBeLessThan(1e-9);
    }
    // After response 360 the internal centers return to the start (full rotation).
    expect(engine.drift.centerA[0]).toBeCloseTo(engine.initial.centerA[0], 12);
    expect(engine.drift.centerA[1]).toBeCloseTo(engine.initial.centerA[1], 12);
    // No feedback in drift, but correctness is recorded.
    for (const r of drift) {
      expect(r.feedbackShown).toBe(false);
      expect(typeof r.correct).toBe("boolean");
    }
    expect(states.filter((s) => s.status === "feedback")).toHaveLength(0);
    expect(states.every((s) => s.feedback === undefined || s.feedback === null)).toBe(true);
    expect(engine.state.screen).toBe(SCREENS.COMPLETE);
    expect(storage.exports.at(-1).name).toBe(`${engine.meta.sessionId}.csv`);
  });

  it("no image repeats within drift; training images may reappear in drift (D5)", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    await passTraining(engine, presenter);
    await runDrift(engine, presenter);
    const trainingIds = engine.trainingRecords.map((r) => r.imageId);
    const driftIds = engine.driftRecords.map((r) => r.imageId);
    expect(new Set(trainingIds).size).toBe(trainingIds.length);
    expect(new Set(driftIds).size).toBe(driftIds.length);
    // With the default configuration the first A training image (the base point) is also the
    // nearest image to the first drift A center; overlap is allowed.
    expect(driftIds.some((id) => trainingIds.includes(id))).toBe(true);
  });

  it("the drift sequence does not depend on how long training took", async () => {
    const short = makeEngine({ seed: 7 });
    await startToTraining(short.engine);
    await passTraining(short.engine, short.presenter);
    await runDrift(short.engine, short.presenter);

    const long = makeEngine({ seed: 7 });
    await startToTraining(long.engine);
    await runTraining(long.engine, long.presenter, [...Array(10).fill(false), ...allCorrect(20)]);
    // 10 wrong then correct: the last-20 window first reaches 16/20 at trial 26.
    expect(long.engine.trainingRecords).toHaveLength(26);
    await runDrift(long.engine, long.presenter);

    const ids = (e) => e.driftRecords.map((r) => `${r.trueGroup}:${r.imageId}`);
    expect(ids(long.engine)).toEqual(ids(short.engine));
  });

  it("supports independent center configuration and alternate trial counts/step sizes", async () => {
    const { engine, presenter } = makeEngine({
      overrides: { "training.initialAngleA": 30, "training.initialAngleB": 210, "drift.numberOfTrials": 100, "drift.degreesPerTrial": 0.5 },
    });
    await startToTraining(engine);
    await passTraining(engine, presenter);
    expect(engine.trainingRecords[0]).toMatchObject({ currentAngleA: 30, currentAngleB: 210 });
    await runDrift(engine, presenter);
    const drift = engine.driftRecords;
    expect(drift).toHaveLength(100);
    expect(drift[0]).toMatchObject({ currentAngleA: 30, currentAngleB: 210 });
    expect(drift[1]).toMatchObject({ currentAngleA: 30.5, currentAngleB: 210.5 });
    expect(drift[99]).toMatchObject({ currentAngleA: 79.5, currentAngleB: 259.5 });
    // B is the reflection of A through the origin at every trial (D3).
    for (const r of drift) {
      expect(r.centerB_pc1).toBeCloseTo(-r.centerA_pc1, 14);
      expect(r.centerB_pc2).toBeCloseTo(-r.centerA_pc2, 14);
    }
  });
});

describe("timing and failures (acceptance check 8)", () => {
  it("a delayed image load does not inflate response time", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    presenter.loadDelayMs = 2000;
    await answer(engine, presenter, correctResponse(engine), 640); // first trial loaded without delay
    await answer(engine, presenter, correctResponse(engine), 750); // loaded with 2 s delay
    expect(engine.records.map((r) => r.responseTimeMs)).toEqual([640, 750]);
  });

  it("a failed load creates no record, does not move centers, and retries the same image", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    await passTraining(engine, presenter);
    await engine.continueToDrift();
    await answer(engine, presenter, correctResponse(engine));
    const expected = engine.pending;
    const centersBefore = [...engine.drift.centerA];
    // Fail the load of drift trial 2.
    presenter.failNext = 1;
    engine.pending = null;
    await engine.runTrial();
    expect(engine.state).toMatchObject({ screen: SCREENS.ERROR, kind: "imageLoad" });
    const pendingAfterFailure = engine.pending;
    expect(pendingAfterFailure.stimulus.imageId).toBe(expected.stimulus.imageId);
    expect(engine.driftRecords).toHaveLength(1);
    expect(engine.drift.centerA).toEqual(centersBefore);
    expect(await engine.respond("A", 99999)).toBe(false); // no timer, no response accepted
    await engine.retry();
    expect(engine.state.status).toBe("responding");
    expect(engine.pending.stimulus.imageId).toBe(expected.stimulus.imageId);
    await answer(engine, presenter, "A");
    expect(engine.driftRecords).toHaveLength(2);
    expect(engine.driftRecords[1].imageLoadFailures).toBe(1);
    expect(engine.driftRecords[1].currentAngleA).toBe(1);
  });

  it("duplicate clicks create only one record", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    presenter.now += 300;
    const first = engine.respond("A", presenter.now);
    const second = engine.respond("B", presenter.now + 5);
    await Promise.all([first, second]);
    expect(await second).toBe(false);
    expect(engine.records).toHaveLength(1);
    expect(engine.records[0].response).toBe("A");
  });

  it("a storage failure is surfaced, nothing advances, and a retry saves the same record once", async () => {
    const storage = new MemoryStorage();
    const { engine, presenter } = makeEngine({ storage });
    await startToTraining(engine);
    const original = storage.saveTrial.bind(storage);
    let failures = 1;
    storage.saveTrial = async (trial) => {
      if (failures-- > 0) throw new StorageError("QUOTA_EXCEEDED", "Browser storage is full.");
      return original(trial);
    };
    await answer(engine, presenter, "Probably A");
    expect(engine.state).toMatchObject({ screen: SCREENS.ERROR, kind: "storage", code: "QUOTA_EXCEEDED" });
    expect(engine.records).toHaveLength(0);
    await engine.retry();
    expect(engine.records).toHaveLength(1);
    const stored = await storage.getSession(engine.meta.sessionId);
    expect(stored.trials).toHaveLength(1);
    expect(stored.trials[0].response).toBe("Probably A");
  });
});

describe("response coding in records (acceptance check 3)", () => {
  it("preserves the four original responses verbatim and derives binary correctness", async () => {
    const { engine, presenter } = makeEngine();
    await startToTraining(engine);
    for (const response of ["A", "Probably A", "Probably B", "B"]) {
      await answer(engine, presenter, response);
    }
    const records = engine.records;
    expect(records.map((r) => r.response)).toEqual(["A", "Probably A", "Probably B", "B"]);
    expect(records.map((r) => r.binaryResponse)).toEqual(["A", "A", "B", "B"]);
    for (const r of records) expect(r.correct).toBe(r.binaryResponse === r.trueGroup);
  });
});

describe("reproducibility and balancing (acceptance check 9)", () => {
  async function fullRun(seed, pattern) {
    const { engine, presenter } = makeEngine({ seed });
    await startToTraining(engine);
    await runTraining(engine, presenter, pattern);
    await runDrift(engine, presenter);
    return engine;
  }

  it("same seed + same responses reproduce groups and stimuli; different seeds differ", async () => {
    const pattern = [false, false, true, false, ...allCorrect(20)];
    const a = await fullRun(99, pattern);
    const b = await fullRun(99, pattern);
    const c = await fullRun(100, pattern);
    const seq = (e) => e.records.map((r) => `${r.phase}:${r.trueGroup}:${r.imageId}`);
    expect(seq(b)).toEqual(seq(a));
    expect(seq(c)).not.toEqual(seq(a));
  });

  it("drift groups differ by at most one; every completed training block is balanced", async () => {
    const e = await fullRun(5, [false, true, false, ...allCorrect(21)]);
    const drift = e.driftRecords;
    const nA = drift.filter((r) => r.trueGroup === "A").length;
    expect(Math.abs(nA - (drift.length - nA))).toBeLessThanOrEqual(1);
    const groups = e.trainingRecords.map((r) => r.trueGroup);
    const size = e.config.randomization.trainingBlockSize;
    for (let i = 0; i + size <= groups.length; i += size) {
      const block = groups.slice(i, i + size);
      expect(block.filter((g) => g === "A")).toHaveLength(size / 2);
    }
  });

  it("odd drift lengths differ by exactly one", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "drift.numberOfTrials": 7 } });
    await startToTraining(engine);
    await passTraining(engine, presenter);
    await runDrift(engine, presenter);
    const nA = engine.driftRecords.filter((r) => r.trueGroup === "A").length;
    expect(Math.abs(2 * nA - 7)).toBe(1);
  });
});

describe("development controls", () => {
  it("finish early saves an incomplete session and never marks the criterion as met", async () => {
    const { engine, presenter, storage } = makeEngine({ developmentMode: true });
    await startToTraining(engine);
    await runTraining(engine, presenter, allCorrect(5));
    await engine.finishEarly();
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE, completionStatus: "incomplete_dev_early_finish" });
    const stored = await storage.getSession(engine.meta.sessionId);
    expect(stored.meta).toMatchObject({
      completionStatus: "incomplete_dev_early_finish",
      terminationReason: "development_early_finish",
      trainingCriterionMet: false,
    });
    expect(storage.exports.at(-1).name).toBe(`${engine.meta.sessionId}_INCOMPLETE.csv`);
  });

  it("partial export is available during the session and marked incomplete", async () => {
    const { engine, presenter, storage } = makeEngine({ developmentMode: true });
    await startToTraining(engine);
    await answer(engine, presenter, "A");
    await engine.exportPartial();
    expect(storage.exports.at(-1).name).toMatch(/_INCOMPLETE\.csv$/);
    expect(storage.exports.at(-1).text).toContain(",true,"); // exportedAsIncomplete
  });

  it("development controls throw in participant mode", async () => {
    const { engine } = makeEngine();
    await expect(engine.finishEarly()).rejects.toThrow();
    await expect(engine.exportPartial()).rejects.toThrow();
  });
});
