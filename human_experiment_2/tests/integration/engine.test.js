// Engine integration tests with fakes: trial structure, continuity, scoring, timing, failures.
import { describe, expect, it } from "vitest";
import { collectNearest } from "../../src/scientific/clusterConstruction.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { StorageError } from "../../src/services/dataStorage/storageContract.js";
import { answer, correctResponse, makeEngine, realPca, runAll, SCREENS, startExperiment, wrongResponse } from "../helpers/engineHarness.js";

const small = { "drift.numberOfTrials": 40, "drift.degreesPerTrial": 9 };

async function fullRun(options, responder) {
  const made = makeEngine(options);
  await startExperiment(made.engine);
  await runAll(made.engine, made.presenter, responder);
  return made;
}

describe("flow", () => {
  it("consent -> instructions -> first trial, with no training phase", async () => {
    const { engine } = makeEngine();
    expect(engine.state.screen).toBe(SCREENS.CONSENT);
    await engine.acceptConsent();
    expect(engine.state.screen).toBe(SCREENS.INSTRUCTIONS);
    expect(engine.meta.consentGiven).toBe(true);
    expect(engine.meta.consentTimestamp).toMatch(/Z$/);
    await engine.startExperiment();
    expect(engine.state).toMatchObject({ screen: SCREENS.TRIAL, status: "responding" });
    expect(engine.pending.trial.trialNumber).toBe(1);
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

  it("ends after exactly numberOfTrials answers", async () => {
    const { engine, storage } = await fullRun({ overrides: small });
    expect(engine.records).toHaveLength(40);
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE, completionStatus: "complete" });
    expect((await storage.getSession(engine.meta.sessionId)).meta).toMatchObject({ completedTrials: 40, terminationReason: "completed_all_trials" });
  });
});

describe("trial structure", () => {
  it("trial 1 compares with the image of center A", async () => {
    const { engine } = await fullRun({ overrides: small });
    const first = engine.records[0];
    expect(first).toMatchObject({ sampleGroup: "A", sampleSourceTrial: null, sampleDistanceToCenter: 0 });
    expect(first.sampleImageIndex).toBe(engine.initial.baseImageIndex);
    const pca = realPca();
    expect(first.sampleImageIndex).toBe(collectNearest(engine.initial.centerA, pca, 1)[0].index);
  });

  it("continuity: every later sample is the previous trial's choice of the scheduled group", async () => {
    const { engine } = await fullRun({ overrides: small });
    const { sampleSchedule } = engine.plan;
    for (let i = 1; i < engine.records.length; i += 1) {
      const [previous, record] = [engine.records[i - 1], engine.records[i]];
      expect(record.sampleGroup).toBe(sampleSchedule[i]);
      expect(record.sampleSourceTrial).toBe(previous.trialNumber);
      expect(record.sampleImageId).toBe(record.sampleGroup === "A" ? previous.optionAImageId : previous.optionBImageId);
    }
    expect(new Set(engine.records.map((r) => r.sampleGroup))).toEqual(new Set(["A", "B"]));
  });

  it("the two choices come from opposite centers: each is near its own group's center and far from the other", async () => {
    const { engine } = await fullRun({ overrides: small });
    for (const r of engine.records) {
      const distance = (x, y, cx, cy) => Math.hypot(x - cx, y - cy);
      expect(distance(r.optionAPc1, r.optionAPc2, r.centerA_pc1, r.centerA_pc2)).toBeCloseTo(r.optionADistanceToCenter, 12);
      expect(distance(r.optionBPc1, r.optionBPc2, r.centerB_pc1, r.centerB_pc2)).toBeCloseTo(r.optionBDistanceToCenter, 12);
      expect(r.optionADistanceToCenter).toBeLessThan(0.2);
      expect(r.optionBDistanceToCenter).toBeLessThan(0.2);
      expect(distance(r.optionAPc1, r.optionAPc2, r.optionBPc1, r.optionBPc2)).toBeGreaterThan(0.5);
    }
  });

  it("no image is selected twice, at the full default length", async () => {
    const { engine } = await fullRun({});
    const images = [engine.records[0].sampleImageId, ...engine.records.flatMap((r) => [r.optionAImageId, r.optionBImageId])];
    expect(images).toHaveLength(721);
    expect(new Set(images).size).toBe(721);
    for (const r of engine.records) expect(new Set([r.sampleImageId, r.optionAImageId, r.optionBImageId]).size).toBe(3);
  });

  it("the sample is shown first, then the blank, then the two choices on their scheduled sides", async () => {
    const { engine, presenter, clock } = makeEngine({ overrides: small });
    await startExperiment(engine);
    const { trial } = engine.pending;
    const url = (s) => engine.imageUrl(s.imageId);
    expect(presenter.events).toEqual([
      ["load", url(trial.sample), url(trial.options.A), url(trial.options.B)],
      ["sample", url(trial.sample)],
      ["hide"],
      ["pair", url(trial.options[trial.leftGroup]), url(trial.options[trial.rightGroup])],
    ]);
    expect(clock.sleeps).toEqual([1000, 500]);
    expect(presenter.enabled).toBe(true);
  });
});

describe("scoring", () => {
  it("correct means choosing the face of the sample's group, with samples from both groups", async () => {
    const { engine } = await fullRun({ overrides: small }, (e, i) => (i % 3 === 0 ? wrongResponse(e) : correctResponse(e)));
    engine.records.forEach((r, i) => {
      expect(r.correct).toBe(i % 3 !== 0);
      expect(r.correct).toBe(r.chosenGroup === r.sampleGroup);
      expect(r.chosenGroup).toBe(r.response === "left" ? r.leftGroup : r.leftGroup === "A" ? "B" : "A");
      expect(r.chosenImageId).toBe(r.chosenGroup === "A" ? r.optionAImageId : r.optionBImageId);
    });
  });

  it("rejects anything but left or right", async () => {
    const { engine, presenter } = makeEngine({ overrides: small });
    await startExperiment(engine);
    await expect(answer(engine, presenter, "A")).rejects.toThrow();
  });
});

describe("drift", () => {
  it("centers move after each answer: nominal angles 0, 1, ... and back to the start", async () => {
    const { engine } = await fullRun({});
    const at = (k) => engine.records[k - 1];
    expect([1, 2, 180, 181, 360].map((k) => [at(k).currentAngleA, at(k).currentAngleB])).toEqual([
      [0, 180],
      [1, 181],
      [179, 359],
      [180, 0],
      [359, 179],
    ]);
  });

  it("supports other starting angles and steps; B stays the reflection of A", async () => {
    const { engine } = await fullRun({ overrides: { "drift.initialAngleA": 30, "drift.initialAngleB": 210, "drift.numberOfTrials": 100, "drift.degreesPerTrial": 0.5 } });
    const r = engine.records;
    expect(r).toHaveLength(100);
    expect(r[0]).toMatchObject({ currentAngleA: 30, currentAngleB: 210 });
    expect(r[1]).toMatchObject({ currentAngleA: 30.5, currentAngleB: 210.5 });
    expect(r[99]).toMatchObject({ currentAngleA: 79.5, currentAngleB: 259.5 });
    for (const t of r) {
      expect(t.centerB_pc1).toBeCloseTo(-t.centerA_pc1, 14);
      expect(t.centerB_pc2).toBeCloseTo(-t.centerA_pc2, 14);
    }
  });
});

describe("timing and failures", () => {
  it("response time runs from the two faces, and excludes loading, the sample and the blank", async () => {
    const { engine, presenter } = makeEngine({ overrides: small });
    await startExperiment(engine);
    presenter.loadDelayMs = 2000;
    await answer(engine, presenter, "left", 640); // first trial loaded without delay
    await answer(engine, presenter, "left", 750); // loaded with a 2 s delay
    expect(engine.records.map((r) => r.responseTimeMs)).toEqual([640, 750]);
  });

  it("records the measured sample and blank durations", async () => {
    const { engine } = await fullRun({ overrides: small });
    expect(engine.records.every((r) => r.sampleDurationMs === 1000 && r.blankDurationMs === 500)).toBe(true);
  });

  it("the click clears the screen, and the next sample waits for the pause while the next images load", async () => {
    const { engine, presenter, clock } = makeEngine({ overrides: { ...small, "timing.interTrialDurationMs": 700 } });
    await startExperiment(engine);
    const sleep = clock.sleep.bind(clock);
    let release;
    clock.sleep = (ms) => (ms === 700 ? new Promise((resolve) => (release = resolve)) : sleep(ms));
    const events = (type) => presenter.events.filter(([name]) => name === type).length;
    const cleared = presenter.cleared;
    const answered = answer(engine, presenter, "left");
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(presenter.cleared).toBe(cleared + 1);
    expect(engine.records).toHaveLength(1); // saved during the pause
    expect(events("load")).toBe(2); // next images loaded during the pause
    expect(events("sample")).toBe(1); // next sample not shown yet
    release();
    await answered;
    expect(events("sample")).toBe(2);
  });

  it("records the measured pause before each sample; trial 1 has none; response time is unaffected", async () => {
    const { engine, clock } = await fullRun({ overrides: small });
    expect(engine.records[0].interTrialDurationMs).toBeNull();
    expect(engine.records.slice(1).every((r) => r.interTrialDurationMs === 500)).toBe(true);
    expect(engine.records.every((r) => r.responseTimeMs === 500)).toBe(true);
    expect(clock.sleeps.slice(0, 5)).toEqual([1000, 500, 500, 1000, 500]);
  });

  it("no response is accepted before the two faces appear", async () => {
    const { engine, presenter } = makeEngine({ overrides: small });
    await engine.acceptConsent();
    const running = engine.startExperiment();
    expect(await engine.respond("left", 0)).toBe(false); // still loading
    await running;
    await answer(engine, presenter, "left");
    expect(engine.records).toHaveLength(1);
  });

  it("a failed load creates no record, shows nothing, keeps the centers, and retries the same trial", async () => {
    const { engine, presenter } = makeEngine({ overrides: small });
    await startExperiment(engine);
    presenter.failNext = 1;
    await answer(engine, presenter, correctResponse(engine));
    expect(engine.state).toMatchObject({ screen: SCREENS.ERROR, kind: "imageLoad" });
    expect(engine.records).toHaveLength(1);
    expect(presenter.events.at(-1)[0]).toBe("pair"); // trial 2 showed nothing
    expect(await engine.respond("left", 99999)).toBe(false);
    await engine.retry();
    expect(engine.state.status).toBe("responding");
    expect(engine.pending.trial.trialNumber).toBe(2);
    await answer(engine, presenter, "left");
    expect(engine.records[1]).toMatchObject({ imageLoadFailures: 1, trialNumber: 2, currentAngleA: 9 });
  });

  it("duplicate clicks create only one record", async () => {
    const { engine, presenter } = makeEngine({ overrides: small });
    await startExperiment(engine);
    presenter.now += 300;
    const first = engine.respond("left", presenter.now);
    const second = engine.respond("right", presenter.now + 5);
    await Promise.all([first, second]);
    expect(await second).toBe(false);
    expect(engine.records[0].response).toBe("left");
  });

  it("a storage failure is surfaced, nothing advances, and a retry saves the same record once", async () => {
    const storage = new MemoryStorage();
    const { engine, presenter } = makeEngine({ storage, overrides: small });
    await startExperiment(engine);
    const original = storage.saveTrial.bind(storage);
    let failures = 1;
    storage.saveTrial = async (trial) => {
      if (failures-- > 0) throw new StorageError("QUOTA_EXCEEDED", "Browser storage is full.");
      return original(trial);
    };
    await answer(engine, presenter, "right");
    expect(engine.state).toMatchObject({ screen: SCREENS.ERROR, kind: "storage", code: "QUOTA_EXCEEDED" });
    expect(engine.records).toHaveLength(0);
    await engine.retry();
    expect(engine.records).toHaveLength(1);
    const stored = await storage.getSession(engine.meta.sessionId);
    expect(stored.trials).toHaveLength(1);
    expect(stored.trials[0].response).toBe("right");
  });
});

describe("reproducibility and balancing", () => {
  const sequence = (engine) => engine.records.map((r) => [r.sampleGroup, r.sampleImageId, r.optionAImageId, r.optionBImageId, r.leftGroup].join(":"));

  it("the same seed gives the same sequence; a different seed differs", async () => {
    const a = await fullRun({ seed: 99, overrides: small });
    const b = await fullRun({ seed: 99, overrides: small });
    const c = await fullRun({ seed: 100, overrides: small });
    expect(sequence(b.engine)).toEqual(sequence(a.engine));
    expect(sequence(c.engine)).not.toEqual(sequence(a.engine));
  });

  it("answers never change the sequence", async () => {
    const right = await fullRun({ seed: 7, overrides: small }, correctResponse);
    const wrong = await fullRun({ seed: 7, overrides: small }, wrongResponse);
    const alwaysLeft = await fullRun({ seed: 7, overrides: small }, () => "left");
    expect(sequence(wrong.engine)).toEqual(sequence(right.engine));
    expect(sequence(alwaysLeft.engine)).toEqual(sequence(right.engine));
  });

  it("sample groups after trial 1 and left sides are balanced, also for odd lengths", async () => {
    for (const n of [40, 41]) {
      const { engine } = await fullRun({ overrides: { "drift.numberOfTrials": n, "drift.degreesPerTrial": 3 } });
      const count = (list) => list.filter((g) => g === "A").length;
      const samples = engine.records.slice(1).map((r) => r.sampleGroup);
      const sides = engine.records.map((r) => r.leftGroup);
      expect(Math.abs(2 * count(samples) - samples.length)).toBeLessThanOrEqual(1);
      expect(Math.abs(2 * count(sides) - sides.length)).toBeLessThanOrEqual(1);
    }
  });

  it("a one-trial session works", async () => {
    const { engine } = await fullRun({ overrides: { "drift.numberOfTrials": 1 } });
    expect(engine.records).toHaveLength(1);
    expect(engine.records[0].sampleGroup).toBe("A");
  });

  it("stores the schedules with the session", async () => {
    const { engine, storage } = await fullRun({ overrides: small });
    const { meta } = await storage.getSession(engine.meta.sessionId);
    expect(meta.randomization.sampleSchedule).toBe(engine.records.map((r) => r.sampleGroup).join(""));
    expect(meta.randomization.leftSchedule).toBe(engine.records.map((r) => r.leftGroup).join(""));
  });
});

describe("development controls", () => {
  it("finish early saves an incomplete session", async () => {
    const { engine, presenter, storage } = makeEngine({ developmentMode: true, overrides: small });
    await startExperiment(engine);
    await answer(engine, presenter, "left");
    await engine.finishEarly();
    expect(engine.state).toMatchObject({ screen: SCREENS.COMPLETE, completionStatus: "incomplete_dev_early_finish" });
    const stored = await storage.getSession(engine.meta.sessionId);
    expect(stored.meta).toMatchObject({ completionStatus: "incomplete_dev_early_finish", terminationReason: "development_early_finish", completedTrials: 1 });
    expect(storage.exports.at(-1).name).toBe(`${engine.meta.sessionId}_INCOMPLETE.csv`);
  });

  it("partial export is available during the session and marked incomplete", async () => {
    const { engine, presenter, storage } = makeEngine({ developmentMode: true, overrides: small });
    await startExperiment(engine);
    await answer(engine, presenter, "left");
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
