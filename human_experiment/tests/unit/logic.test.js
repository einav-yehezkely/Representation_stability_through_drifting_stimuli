// Unit tests: response coding, training criterion, randomization, configuration, identity.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { findConfigProblems, resolveEffectiveConfig } from "../../src/config/validateConfig.js";
import { isCorrect, RESPONSES, toBinary } from "../../src/experiment/responseCoding.js";
import { createDriftSchedule, SeededRng, TrainingSchedule } from "../../src/experiment/seededRandomization.js";
import { resolveParticipantId } from "../../src/experiment/session.js";
import { evaluateCriterion } from "../../src/experiment/trainingPhase.js";
import { nominalDriftAngle } from "../../src/scientific/trajectory.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.resolve(here, "..", "..", "src");

describe("response coding (PRD §5)", () => {
  it("has the four responses in the fixed order", () => {
    expect(RESPONSES).toEqual(["A", "Probably A", "Probably B", "B"]);
  });
  it("maps responses to binary and ignores confidence for correctness", () => {
    expect(RESPONSES.map(toBinary)).toEqual(["A", "A", "B", "B"]);
    expect(isCorrect(toBinary("Probably A"), "A")).toBe(true);
    expect(isCorrect(toBinary("Probably B"), "A")).toBe(false);
    expect(() => toBinary("C")).toThrow();
  });
});

describe("training criterion (PRD §6)", () => {
  const cfg = experimentConfig.training;
  const hist = (bits) => bits.map((c) => ({ correct: Boolean(c) }));
  it("never evaluates an incomplete window", () => {
    expect(evaluateCriterion(hist(Array(19).fill(1)), cfg)).toMatchObject({ eligible: false, passed: false, accuracy: null });
  });
  it("16/20 passes, 15/20 fails", () => {
    expect(evaluateCriterion(hist([0, 0, 0, 0, ...Array(16).fill(1)]), cfg)).toMatchObject({ passed: true, accuracy: 0.8 });
    expect(evaluateCriterion(hist([0, 0, 0, 0, 0, ...Array(15).fill(1)]), cfg)).toMatchObject({ passed: false, accuracy: 0.75 });
  });
  it("uses the last 20 trials, not cumulative accuracy", () => {
    const h = hist([...Array(40).fill(1), ...Array(5).fill(0), ...Array(15).fill(1)]); // 55/60 cumulative
    expect(evaluateCriterion(h, cfg)).toMatchObject({ passed: false, accuracy: 0.75 });
  });
  it("requires both minTrials and the window", () => {
    const c = { minTrials: 30, accuracyWindow: 10, accuracyThreshold: 0.8 };
    expect(evaluateCriterion(hist(Array(29).fill(1)), c).passed).toBe(false);
    expect(evaluateCriterion(hist(Array(30).fill(1)), c).passed).toBe(true);
  });
});

describe("nominal drift angles (PRD §7 table)", () => {
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

describe("seeded randomization (PRD §8)", () => {
  it("is deterministic per seed and stream; streams are independent", () => {
    const seq = (seed, stream) => {
      const rng = new SeededRng(seed, stream);
      return Array.from({ length: 5 }, () => rng.nextUint32());
    };
    expect(seq(1, "driftSchedule")).toEqual(seq(1, "driftSchedule"));
    expect(seq(1, "driftSchedule")).not.toEqual(seq(2, "driftSchedule"));
    expect(seq(1, "driftSchedule")).not.toEqual(seq(1, "trainingSchedule"));
  });
  it("produces uniform-looking floats in [0, 1)", () => {
    const rng = new SeededRng(42, "x");
    const values = Array.from({ length: 20000 }, () => rng.next());
    expect(Math.min(...values)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...values)).toBeLessThan(1);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    expect(Math.abs(mean - 0.5)).toBeLessThan(0.01);
  });
  it("drift schedule is balanced (|A-B| <= 1) and shuffled", () => {
    for (const n of [1, 2, 7, 360]) {
      const s = createDriftSchedule(n, new SeededRng(3, "driftSchedule"));
      const a = s.filter((g) => g === "A").length;
      expect(s).toHaveLength(n);
      expect(Math.abs(a - (n - a))).toBeLessThanOrEqual(1);
    }
    const s = createDriftSchedule(360, new SeededRng(3, "driftSchedule"));
    expect(s.slice(0, 180).every((g) => g === "A")).toBe(false);
  });
  it("training blocks are balanced and generated lazily", () => {
    const t = new TrainingSchedule(4, new SeededRng(9, "trainingSchedule"));
    const groups = Array.from({ length: 10 }, () => t.next());
    expect(t.blocks).toHaveLength(3);
    for (const block of t.blocks) expect(block.filter((g) => g === "A")).toHaveLength(2);
    expect(groups).toEqual(t.blocks.flat().slice(0, 10));
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

describe("configuration validation (PRD §10)", () => {
  it("accepts the default configuration", () => {
    expect(findConfigProblems(experimentConfig, { pcaCount: 11817 })).toEqual([]);
  });
  it("rejects centers that are not 180° apart (decision D3)", () => {
    const bad = resolveEffectiveConfig(experimentConfig, { "training.initialAngleB": 170 });
    expect(findConfigProblems(bad).join(" ")).toMatch(/180°/);
    const ok = resolveEffectiveConfig(experimentConfig, { "training.initialAngleA": 300, "training.initialAngleB": 120 });
    expect(findConfigProblems(ok)).toEqual([]);
  });
  it("rejects invalid counts, thresholds and angles", () => {
    const cases = {
      "training.minTrials": 0,
      "training.accuracyWindow": 2.5,
      "training.accuracyThreshold": 1.2,
      "drift.numberOfTrials": -1,
      "drift.degreesPerTrial": Number.NaN,
      "training.initialAngleA": Number.POSITIVE_INFINITY,
      "randomization.trainingBlockSize": 3,
    };
    for (const [key, value] of Object.entries(cases)) {
      const problems = findConfigProblems(resolveEffectiveConfig(experimentConfig, { [key]: value }));
      expect(problems.length, key).toBeGreaterThan(0);
    }
  });
  it("records overrides without mutating the base configuration", () => {
    const eff = resolveEffectiveConfig(experimentConfig, { "drift.numberOfTrials": 10 });
    expect(eff.drift.numberOfTrials).toBe(10);
    expect(experimentConfig.drift.numberOfTrials).toBe(360);
    expect(Object.isFrozen(eff.drift)).toBe(true);
    expect(() => resolveEffectiveConfig(experimentConfig, { "drift.nope": 1 })).toThrow();
  });
});

describe("participant identity (PRD §11)", () => {
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
