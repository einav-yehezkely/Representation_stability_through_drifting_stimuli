// Trajectory membership used by the researcher visualization.
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { initialCenters } from "../../src/scientific/trajectory.js";
import { driftCandidateIndices, sourceRotationSequence } from "../../src/visualization/trajectoryMembership.js";
import { loadParityFixture, loadRealPca } from "../helpers/loadRealPca.js";

const pca = loadRealPca();
const fx = loadParityFixture();

describe("trajectory membership", () => {
  it("source rotation sequence matches generate_rotation_sequence", () => {
    const start = initialCenters(pca, experimentConfig);
    const seq = sourceRotationSequence(pca, start.centerA);
    expect(seq.map((s) => s.index)).toEqual(fx.rotationSequence.map((s) => s.index));
  });
  it("drift candidates are the nearest images along the configured drift (251 by default)", () => {
    const members = driftCandidateIndices(pca, experimentConfig);
    expect(members).toHaveLength(251);
    const driftChoices = new Set(fx.drift[0].trials.slice(0, 1).map((t) => t.index));
    for (const i of driftChoices) expect(members).toContain(i);
  });
});
