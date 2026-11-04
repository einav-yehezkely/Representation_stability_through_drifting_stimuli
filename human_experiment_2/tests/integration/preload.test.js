// Image preloading and the sequence planner: the plan is exactly what the engine shows, and
// preloading never changes it.
import { describe, expect, it } from "vitest";
import { planSession } from "../../src/experiment/sequencePlanner.js";
import { answer, correctResponse, makeEngine, realPca, runAll, SCREENS, startExperiment } from "../helpers/engineHarness.js";

/** The fake presenter records every preload call. */
function withPreloadLog(presenter) {
  presenter.preloads = [];
  presenter.preload = (urls) => presenter.preloads.push(urls);
  return presenter;
}

describe("planSession", () => {
  it("predicts every image of a real session", async () => {
    const { engine, presenter, config } = makeEngine({ overrides: { "drift.numberOfTrials": 60, "drift.degreesPerTrial": 6 }, seed: 99 });
    await startExperiment(engine);
    await runAll(engine, presenter);
    expect(engine.state.screen).toBe(SCREENS.COMPLETE);
    const plan = planSession({ pca: realPca(), config, sessionSeed: 99 });
    expect(plan.trials.map((t) => [t.sampleGroup, t.sample.imageId, t.options.A.imageId, t.options.B.imageId, t.leftGroup, t.snapshot.nominalAngleA])).toEqual(
      engine.records.map((r) => [r.sampleGroup, r.sampleImageId, r.optionAImageId, r.optionBImageId, r.leftGroup, r.currentAngleA]),
    );
  });

  it("limit plans only the first trials, identically", () => {
    const { config } = makeEngine();
    const args = { pca: realPca(), config, sessionSeed: 3 };
    const ids = (trials) => trials.map((t) => [t.sample.imageId, t.options.A.imageId, t.options.B.imageId]);
    const part = planSession({ ...args, limit: 12 }).trials;
    expect(part).toHaveLength(12);
    expect(ids(part)).toEqual(ids(planSession(args).trials.slice(0, 12)));
  });
});

describe("engine preloading", () => {
  it("announces the images of a trial before they are needed", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "drift.numberOfTrials": 12 }, seed: 5 });
    withPreloadLog(presenter);
    await engine.acceptConsent();
    expect(presenter.preloads).toHaveLength(1); // at the instructions screen
    await engine.startExperiment();
    for (let i = 0; engine.state.screen === SCREENS.TRIAL; i += 1) {
      const announced = presenter.preloads.at(-1);
      for (const url of engine.trialUrls(engine.pending.trial)) expect(announced).toContain(url);
      const before = presenter.preloads.length;
      await answer(engine, presenter, correctResponse(engine));
      expect(presenter.preloads.length).toBe(before + 1);
    }
    expect(presenter.preloads.at(-1)).toEqual([]); // nothing after the last trial
    expect(new Set(presenter.preloads[0]).size).toBe(presenter.preloads[0].length);
  });

  it("preloading does not change the session (same records as without a preloader)", async () => {
    const run = async (preload) => {
      const { engine, presenter } = makeEngine({ overrides: { "drift.numberOfTrials": 20 }, seed: 31 });
      if (preload) withPreloadLog(presenter);
      await startExperiment(engine);
      await runAll(engine, presenter);
      return engine.records.map((r) => [r.trialNumber, r.sampleGroup, r.sampleImageId, r.optionAImageId, r.optionBImageId, r.currentAngleA]);
    };
    expect(await run(true)).toEqual(await run(false));
  });

  it("ui.preloadCount 0 disables preloading", async () => {
    const { engine, presenter } = makeEngine({ overrides: { "ui.preloadCount": 0 } });
    withPreloadLog(presenter);
    await startExperiment(engine);
    await answer(engine, presenter, "left");
    expect(presenter.preloads).toEqual([]);
  });
});
