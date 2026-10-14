// Image preloading and the sequence planner: lookahead must predict exactly what the engine shows
// and must never change it.
import { describe, expect, it } from "vitest";
import { planSession } from "../../src/experiment/sequencePlanner.js";
import { createRngFactory, TrainingSchedule } from "../../src/experiment/seededRandomization.js";
import { answer, correctResponse, makeEngine, realPca, runTraining, SCREENS, startToTraining, wrongResponse } from "../helpers/engineHarness.js";

/** The fake presenter records every preload call. */
function withPreloadLog(presenter) {
  presenter.preloads = [];
  presenter.preload = (urls) => presenter.preloads.push(urls);
  return presenter;
}

async function runSession(engine, presenter, pattern) {
  await startToTraining(engine);
  await runTraining(engine, presenter, pattern);
  await engine.continueToDrift();
  while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, correctResponse(engine));
}

describe("lookahead primitives", () => {
  it("TrainingSchedule.peek returns the upcoming groups without consuming them", () => {
    const schedule = new TrainingSchedule(4, createRngFactory(7)("trainingSchedule"));
    schedule.next();
    const draws = schedule.rng.draws;
    const peeked = schedule.peek(9);
    expect(schedule.rng.draws).toBe(draws);
    expect(schedule.position).toBe(1);
    expect(Array.from({ length: 9 }, () => schedule.next())).toEqual(peeked);
  });
});

describe("planSession", () => {
  it("predicts every image of a real session, whatever the training length", async () => {
    const overrides = { "drift.numberOfTrials": 60, "drift.degreesPerTrial": 6 };
    for (const pattern of [Array(20).fill(true), [...Array(13).fill(false), ...Array(20).fill(true)]]) {
      const { engine, presenter, config } = makeEngine({ overrides, seed: 99 });
      await runSession(engine, presenter, pattern);
      expect(engine.state.screen).toBe(SCREENS.COMPLETE);
      const plan = planSession({ pca: realPca(), config, sessionSeed: 99, trainingTrials: engine.trainingRecords.length });
      expect(plan.training.map((t) => [t.group, t.stimulus.imageId])).toEqual(engine.trainingRecords.map((r) => [r.trueGroup, r.imageId]));
      expect(plan.drift.map((t) => [t.group, t.stimulus.imageId, t.snapshot.nominalAngleA])).toEqual(
        engine.driftRecords.map((r) => [r.trueGroup, r.imageId, r.currentAngleA]),
      );
    }
  });
});

describe("engine preloading", () => {
  it("announces the next image before it is shown, in training, at the transition and in drift", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "drift.numberOfTrials": 12 }, seed: 5 });
    withPreloadLog(presenter);
    await engine.acceptConsent();
    expect(presenter.preloads).toHaveLength(1); // at the instructions screen
    let announced = presenter.preloads.at(-1);
    await engine.startTraining();
    for (let i = 0; ; i += 1) {
      expect(presenter.shown.at(-1)).toBe(announced[0]);
      if (engine.state.screen !== SCREENS.TRIAL) break;
      const before = presenter.preloads.length;
      await answer(engine, presenter, i < 3 ? wrongResponse(engine) : correctResponse(engine));
      expect(presenter.preloads.length).toBe(before + 1);
      announced = presenter.preloads.at(-1);
      if (engine.state.screen === SCREENS.TRANSITION) {
        await engine.continueToDrift();
      }
      if (engine.state.screen === SCREENS.COMPLETE) {
        expect(announced).toEqual([]); // nothing after the last drift trial
        break;
      }
    }
    expect(engine.driftRecords).toHaveLength(12);
    expect(presenter.preloads.every((urls) => urls.length <= 3)).toBe(true);
  });

  it("preloading does not change the session (same records as without a preloader)", async () => {
    const run = async (preload) => {
      const { engine, presenter } = makeEngine({ overrides: { "drift.numberOfTrials": 20 }, seed: 31 });
      if (preload) withPreloadLog(presenter);
      await runSession(engine, presenter, [false, true, false, ...Array(20).fill(true)]);
      return engine.records.map((r) => [r.trialNumber, r.phase, r.trueGroup, r.imageId, r.currentAngleA]);
    };
    expect(await run(true)).toEqual(await run(false));
  });

  it("ui.preloadCount 0 disables preloading", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "ui.preloadCount": 0 } });
    withPreloadLog(presenter);
    await engine.acceptConsent();
    await engine.startTraining();
    await answer(engine, presenter, "A");
    expect(presenter.preloads).toEqual([]);
  });
});
