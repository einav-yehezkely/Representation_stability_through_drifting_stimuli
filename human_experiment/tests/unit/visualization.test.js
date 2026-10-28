// Trajectory membership used by the researcher visualization.
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { driftCandidateIndices } from "../../src/visualization/trajectoryMembership.js";
import { loadParityFixture, loadRealPca } from "../helpers/loadRealPca.js";

const pca = loadRealPca();
const fx = loadParityFixture();

describe("trajectory membership", () => {
  it("drift candidates are the nearest images along the configured drift (251 by default)", () => {
    const members = driftCandidateIndices(pca, experimentConfig);
    expect(members).toHaveLength(251); // data-dependent (pca_top2_filtered_female_vgg_1.csv)
    const driftChoices = new Set(fx.drift[0].trials.slice(0, 1).map((t) => t.index));
    for (const i of driftChoices) expect(members).toContain(i);
  });
});
