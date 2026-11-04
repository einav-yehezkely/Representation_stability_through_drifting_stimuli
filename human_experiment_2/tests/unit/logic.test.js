// Unit tests: response coding, randomization, configuration, identity.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { findConfigProblems, resolveEffectiveConfig } from "../../src/config/validateConfig.js";
import { chosenGroup, isCorrect, RESPONSES } from "../../src/experiment/responseCoding.js";
import { createBalancedSchedule, SeededRng } from "../../src/experiment/seededRandomization.js";
import { resolveParticipantId } from "../../src/experiment/session.js";
import { nominalDriftAngle } from "../../src/scientific/trajectory.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(here, "..", "..", "src");

describe("response coding", () => {
  it("has the two sides", () => {
    expect(RESPONSES).toEqual(["left", "right"]);
  });
  it("derives the chosen group from the side assignment; correct means the sample's group", () => {
    const trial = { leftGroup: "B", rightGroup: "A" };
    expect(chosenGroup("left", trial)).toBe("B");
    expect(chosenGroup("right", trial)).toBe("A");
    expect(() => chosenGroup("A", trial)).toThrow();
    expect(isCorrect("A", "A")).toBe(true);
    expect(isCorrect("B", "B")).toBe(true);
    expect(isCorrect("B", "A")).toBe(false);
  });
});

describe("nominal drift angles", () => {
  it("matches the default progression", () => {
    expect([1, 2, 180, 181, 360].map((k) => [nominalDriftAngle(0, k, 1), nominalDriftAngle(180, k, 1)])).toEqual([
      [0, 180],
      [1, 181],
      [179, 359],
      [180, 0],
      [359, 179],
    ]);
  });
});

describe("seeded randomization", () => {
  it("is deterministic per seed and stream; streams are independent", () => {
    const seq = (seed, stream) => {
      const rng = new SeededRng(seed, stream);
      return Array.from({ length: 5 }, () => rng.nextUint32());
    };
    expect(seq(1, "sampleSchedule")).toEqual(seq(1, "sampleSchedule"));
    expect(seq(1, "sampleSchedule")).not.toEqual(seq(2, "sampleSchedule"));
    expect(seq(1, "sampleSchedule")).not.toEqual(seq(1, "sideSchedule"));
  });
  it("produces uniform-looking floats in [0, 1)", () => {
    const rng = new SeededRng(42, "x");
    const values = Array.from({ length: 20000 }, () => rng.next());
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    expect(Math.abs(mean - 0.5)).toBeLessThan(0.01);
  });
  it("balanced schedules have |A-B| <= 1 and are shuffled", () => {
    for (const n of [1, 2, 7, 360]) {
      const s = createBalancedSchedule(n, new SeededRng(3, "sampleSchedule"));
      const a = s.filter((g) => g === "A").length;
      expect(s).toHaveLength(n);
      expect(Math.abs(a - (n - a))).toBeLessThanOrEqual(1);
    }
    const s = createBalancedSchedule(360, new SeededRng(3, "sampleSchedule"));
    expect(s.slice(0, 180).every((g) => g === "A")).toBe(false);
  });
  it("experiment code never uses Math.random()", () => {
    const files = [];
    const walk = (dir) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (entry.name.endsWith(".js")) files.push(full);
      }
    };
    walk(srcDir);
    const offenders = files.filter((f) => /Math\.random\s*\(/.test(fs.readFileSync(f, "utf8")));
    expect(offenders).toEqual([]);
  });
});

describe("fixed session seed", () => {
  it("the configuration fixes one valid seed; other policies and invalid seeds are rejected", () => {
    expect(experimentConfig.randomization.seedPolicy).toBe("fixed");
    expect(Number.isInteger(experimentConfig.randomization.fixedSeed)).toBe(true);
    const withRandomization = (patch) => ({ ...experimentConfig, randomization: { ...experimentConfig.randomization, ...patch } });
    expect(findConfigProblems(withRandomization({ seedPolicy: "crypto-random-uint32" })).join(" ")).toMatch(/seedPolicy/);
    for (const fixedSeed of [undefined, -1, 1.5, 2 ** 32, "7"]) {
      expect(findConfigProblems(withRandomization({ fixedSeed })).join(" ")).toMatch(/fixedSeed/);
    }
  });
});

describe("configuration validation", () => {
  it("accepts the default configuration", () => {
    expect(findConfigProblems(experimentConfig, { pcaCount: 11817 })).toEqual([]);
  });
  it("rejects centers that are not 180° apart", () => {
    const bad = resolveEffectiveConfig(experimentConfig, { "drift.initialAngleB": 170 });
    expect(findConfigProblems(bad).join(" ")).toMatch(/180°/);
    const ok = resolveEffectiveConfig(experimentConfig, { "drift.initialAngleA": 300, "drift.initialAngleB": 120 });
    expect(findConfigProblems(ok)).toEqual([]);
  });
  it("rejects invalid counts, durations and angles", () => {
    const cases = {
      "timing.sampleDurationMs": -1,
      "timing.blankDurationMs": Number.NaN,
      "drift.numberOfTrials": -1,
      "drift.degreesPerTrial": Number.NaN,
      "drift.initialAngleA": Number.POSITIVE_INFINITY,
    };
    for (const [key, value] of Object.entries(cases)) {
      const problems = findConfigProblems(resolveEffectiveConfig(experimentConfig, { [key]: value }));
      expect(problems.length, key).toBeGreaterThan(0);
    }
  });
  it("needs two new images per trial and one more for the first sample", () => {
    const withTrials = (n) => resolveEffectiveConfig(experimentConfig, { "drift.numberOfTrials": n });
    expect(findConfigProblems(withTrials(5), { pcaCount: 11 })).toEqual([]);
    expect(findConfigProblems(withTrials(6), { pcaCount: 11 }).join(" ")).toMatch(/more images/);
  });
  it("records overrides without mutating the base configuration", () => {
    const eff = resolveEffectiveConfig(experimentConfig, { "drift.numberOfTrials": 10 });
    expect(eff.drift.numberOfTrials).toBe(10);
    expect(experimentConfig.drift.numberOfTrials).toBe(360);
    expect(Object.isFrozen(eff.drift)).toBe(true);
    expect(() => resolveEffectiveConfig(experimentConfig, { "drift.nope": 1 })).toThrow();
  });
});

describe("participant identity", () => {
  it("uses a non-empty PROLIFIC_PID", () => {
    expect(resolveParticipantId("?PROLIFIC_PID=abc123&STUDY_ID=x")).toEqual({ participantId: "abc123", participantIdSource: "prolific_url" });
  });
  it("generates an anonymous id otherwise", () => {
    for (const search of ["", "?PROLIFIC_PID=", "?PROLIFIC_PID=%20%20"]) {
      const id = resolveParticipantId(search);
      expect(id.participantIdSource).toBe("generated_anonymous");
      expect(id.participantId).toMatch(/^anon-[0-9a-f]{16}$/);
    }
  });
});
