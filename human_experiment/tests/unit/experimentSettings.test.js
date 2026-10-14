// Experiment settings set on the admin page: derivations, validation, loading, effect on sessions.
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import {
  applySettings,
  findSettingsProblems,
  loadExperimentSettings,
  oppositeAngle,
  saveLocalSettings,
  SettingsError,
  settingsFromConfig,
  trialsForFullRotation,
} from "../../src/config/experimentSettings.js";
import { answer, makeEngine, runTraining, SCREENS, startToTraining, correctResponse } from "../helpers/engineHarness.js";

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

const settings = (patch = {}) => ({ ...settingsFromConfig(experimentConfig), ...patch });

describe("derived values", () => {
  it("full rotation trial count and opposite angle", () => {
    expect(trialsForFullRotation(1)).toBe(360);
    expect(trialsForFullRotation(2)).toBe(180);
    expect(trialsForFullRotation(0.5)).toBe(720);
    expect(trialsForFullRotation(7)).toBe(51); // 360 / 7 = 51.4 -> rounded
    expect(trialsForFullRotation(0)).toBeNull();
    expect(trialsForFullRotation(-1)).toBeNull();
    expect(oppositeAngle(0)).toBe(180);
    expect(oppositeAngle(270)).toBe(90);
    expect(oppositeAngle(-30)).toBe(150);
  });

  it("defaults are the configuration values", () => {
    expect(settingsFromConfig(experimentConfig)).toEqual({
      "training.initialAngleA": 0,
      "training.initialAngleB": 180,
      "drift.degreesPerTrial": 1,
      "drift.numberOfTrials": 360,
    });
  });
});

describe("validation", () => {
  it("accepts valid settings and rejects broken ones", () => {
    expect(findSettingsProblems(settings({ "training.initialAngleA": 45, "training.initialAngleB": 225, "drift.degreesPerTrial": 2, "drift.numberOfTrials": 180 }), experimentConfig, { pcaCount: 11817 })).toEqual([]);
    expect(findSettingsProblems(settings({ "training.initialAngleB": 200 }), experimentConfig).join(" ")).toMatch(/180°/);
    expect(findSettingsProblems(settings({ "drift.degreesPerTrial": 0 }), experimentConfig).join(" ")).toMatch(/greater than 0/);
    expect(findSettingsProblems(settings({ "drift.numberOfTrials": 2.5 }), experimentConfig).join(" ")).toMatch(/numberOfTrials/);
    expect(findSettingsProblems(settings({ "drift.numberOfTrials": 20000 }), experimentConfig, { pcaCount: 11817 }).join(" ")).toMatch(/exceeds/);
    expect(findSettingsProblems(settings({ "training.initialAngleA": Number.NaN }), experimentConfig).join(" ")).toMatch(/must be a number/);
  });

  it("applySettings changes only the four values and never mutates experimentConfig", () => {
    const config = applySettings(experimentConfig, settings({ "training.initialAngleA": 90, "training.initialAngleB": 270, "drift.degreesPerTrial": 3, "drift.numberOfTrials": 120 }));
    expect(config.training).toMatchObject({ initialAngleA: 90, initialAngleB: 270, minTrials: experimentConfig.training.minTrials });
    expect(config.drift).toEqual({ numberOfTrials: 120, degreesPerTrial: 3 });
    expect(experimentConfig.training.initialAngleA).toBe(0);
    expect(applySettings(experimentConfig, null)).toEqual(JSON.parse(JSON.stringify(experimentConfig)));
  });
});

describe("loading", () => {
  it("local (dev server): saved settings round-trip; nothing saved means defaults", async () => {
    const ls = new MapStorage();
    expect(await loadExperimentSettings({ source: "local", localStorage: ls })).toEqual({ values: null, updatedAt: null, source: "local" });
    saveLocalSettings(settings({ "drift.degreesPerTrial": 2, "drift.numberOfTrials": 180 }), { localStorage: ls, now: () => new Date("2026-10-14T10:00:00Z") });
    const loaded = await loadExperimentSettings({ source: "local", localStorage: ls });
    expect(loaded.values["drift.numberOfTrials"]).toBe(180);
    expect(loaded.updatedAt).toBe("2026-10-14T10:00:00.000Z");
  });

  it("server: reads save.php?settings=1 and fails loudly instead of silently using defaults", async () => {
    const urls = [];
    const ok = async (url) => (urls.push(url), { ok: true, status: 200, json: async () => ({ ok: true, settings: { values: settings({ "drift.numberOfTrials": 90, "drift.degreesPerTrial": 4 }), updatedAt: "t" } }) });
    const loaded = await loadExperimentSettings({ source: "server", endpoint: "save.php", fetch: ok });
    expect(urls).toEqual(["save.php?settings=1"]);
    expect(loaded).toMatchObject({ source: "server", updatedAt: "t" });
    expect(loaded.values["drift.numberOfTrials"]).toBe(90);
    const down = async () => {
      throw new Error("offline");
    };
    await expect(loadExperimentSettings({ source: "server", endpoint: "save.php", fetch: down })).rejects.toBeInstanceOf(SettingsError);
  });
});

describe("a session with saved settings", () => {
  it("starts at the saved angles, moves by the saved step for the saved number of trials", async () => {
    const values = settings({ "training.initialAngleA": 90, "training.initialAngleB": 270, "drift.degreesPerTrial": 30, "drift.numberOfTrials": 12 });
    const overrides = values; // main.js applies saved settings the same way (resolveEffectiveConfig)
    const { engine, presenter } = makeEngine({ overrides });
    await startToTraining(engine);
    await runTraining(engine, presenter, Array(20).fill(true));
    expect(engine.trainingRecords.every((r) => r.currentAngleA === 90 && r.currentAngleB === 270)).toBe(true);
    await engine.continueToDrift();
    while (engine.state.screen === SCREENS.TRIAL) await answer(engine, presenter, correctResponse(engine));
    expect(engine.driftRecords.map((r) => r.currentAngleA)).toEqual([90, 120, 150, 180, 210, 240, 270, 300, 330, 0, 30, 60]);
    expect(engine.driftRecords.map((r) => r.currentAngleB)).toEqual([270, 300, 330, 0, 30, 60, 90, 120, 150, 180, 210, 240]);
  });

  it("the engine stores the applied settings in the session metadata", () => {
    const { engine } = makeEngine();
    expect(engine.meta).toHaveProperty("experimentSettings", null);
  });
});
