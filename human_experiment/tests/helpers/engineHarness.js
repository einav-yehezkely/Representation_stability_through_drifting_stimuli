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

/** Monotonic fake time: presenter onset and response times are explicit. */
export class FakePresenter {
  constructor() {
    this.now = 1000;
    this.shown = [];
    this.failNext = 0;
    this.loadDelayMs = 0;
    this.enabled = false;
    this.cleared = 0;
  }

  async show(url) {
    this.enabled = false;
    if (this.failNext > 0) {
      this.failNext -= 1;
      throw new ImageLoadError(url, new Error("fake load failure"));
    }
    this.now += this.loadDelayMs; // loading time passes before onset
    this.shown.push(url);
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

export function fakeClock() {
  let t = Date.parse("2026-10-07T10:00:00.000Z");
  return {
    sleeps: [],
    async sleep(ms) {
      this.sleeps.push(ms);
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
  const clock = fakeClock();
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

/** Respond to the current trial. `policy(pending)` returns the response label. */
export async function answer(engine, presenter, response, rtMs = 500) {
  presenter.now += rtMs;
  return engine.respond(response, presenter.now);
}

export const correctResponse = (engine) => engine.pending.group;
export const wrongResponse = (engine) => (engine.pending.group === "A" ? "B" : "A");

/** Answer training trials following `pattern` (array of booleans: correct?) until transition. */
export async function runTraining(engine, presenter, pattern) {
  for (const ok of pattern) {
    if (engine.state.screen !== SCREENS.TRIAL) break;
    await answer(engine, presenter, ok ? correctResponse(engine) : wrongResponse(engine));
  }
}

export async function startToTraining(engine) {
  await engine.acceptConsent();
  await engine.startTraining();
}

export { SCREENS };
