// Test harness: engine with a fake presenter, fake clock and in-memory storage.
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { resolveEffectiveConfig } from "../../src/config/validateConfig.js";
import { ExperimentEngine, ImageLoadError, SCREENS } from "../../src/experiment/experimentEngine.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { loadRealPca } from "./loadRealPca.js";

let cachedPca;
export function realPca() {
  cachedPca ??= loadRealPca();
  return cachedPca;
}

/** Fake time: loading and the clock's sleeps advance `now`; response times are explicit. */
export class FakePresenter {
  constructor() {
    this.now = 1000;
    this.events = [];
    this.failNext = 0;
    this.loadDelayMs = 0;
    this.enabled = false;
    this.cleared = 0;
  }

  async load(urls) {
    this.enabled = false;
    if (this.failNext > 0) {
      this.failNext -= 1;
      throw new ImageLoadError(urls[0], new Error("fake load failure"));
    }
    this.now += this.loadDelayMs;
    this.events.push(["load", ...urls]);
  }

  async showSample(url) {
    this.events.push(["sample", url]);
    return this.now;
  }

  async hideSample() {
    this.events.push(["hide"]);
    return this.now;
  }

  async showPair(left, right) {
    this.events.push(["pair", left, right]);
    this.enabled = true;
    return this.now; // onset
  }

  disableResponses() {
    this.enabled = false;
  }

  clear() {
    this.cleared += 1;
  }
}

export function fakeClock(onSleep = () => {}) {
  let t = Date.parse("2026-10-07T10:00:00.000Z");
  return {
    sleeps: [],
    async sleep(ms) {
      this.sleeps.push(ms);
      onSleep(ms);
    },
    wallNow() {
      t += 1;
      return new Date(t);
    },
  };
}

export function makeEngine({ overrides = {}, seed = 12345, storage = new MemoryStorage(), developmentMode = false } = {}) {
  const config = resolveEffectiveConfig(experimentConfig, overrides);
  const presenter = new FakePresenter();
  const clock = fakeClock((ms) => (presenter.now += ms));
  const engine = new ExperimentEngine({
    config,
    developmentMode,
    developmentOverrides: overrides,
    pca: realPca(),
    storage,
    presenter,
    clock,
    sessionId: `test-session-${seed}`,
    sessionSeed: seed,
    identity: { participantId: "P1", participantIdSource: "prolific_url" },
  });
  return { engine, presenter, storage, clock, config };
}

/** Answer the current trial with "left" or "right". */
export async function answer(engine, presenter, response, rtMs = 500) {
  presenter.now += rtMs;
  return engine.respond(response, presenter.now);
}

const sideOf = (trial, group) => (trial.leftGroup === group ? "left" : "right");
export const correctResponse = (engine) => sideOf(engine.pending.trial, engine.pending.trial.sampleGroup);
export const wrongResponse = (engine) => (correctResponse(engine) === "left" ? "right" : "left");

export async function startExperiment(engine) {
  await engine.acceptConsent();
  await engine.startExperiment();
}

/** Answer every remaining trial; `responder(engine, index)` returns the side. */
export async function runAll(engine, presenter, responder = correctResponse) {
  for (let i = 0; engine.state.screen === SCREENS.TRIAL; i += 1) await answer(engine, presenter, responder(engine, i));
}

export { SCREENS };
