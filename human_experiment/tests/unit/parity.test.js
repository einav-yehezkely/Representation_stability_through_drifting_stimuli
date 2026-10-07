// Acceptance check 2: the JavaScript port reproduces the source experiment's trajectory,
// cluster construction, rotation and selection on the real PCA data. Reference values are
// produced by scripts/reference/make_parity_fixtures.py from verbatim copies of the source
// functions (scripts/reference/source_functions.py).
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { collectNearest } from "../../src/scientific/clusterConstruction.js";
import { selectStimulus } from "../../src/scientific/stimulusSelector.js";
import { angleDeg, createBasePointIndex, initialCenters, rotateVector } from "../../src/scientific/trajectory.js";
import { loadParityFixture, loadRealPca } from "../helpers/loadRealPca.js";

const pca = loadRealPca();
const fx = loadParityFixture();
const sci = experimentConfig.scientific;
const TOL = 1e-12;

function expectVecClose(actual, expected, tol = TOL) {
  expect(Math.abs(actual[0] - expected[0])).toBeLessThanOrEqual(tol);
  expect(Math.abs(actual[1] - expected[1])).toBeLessThanOrEqual(tol);
}

describe("parity with the source experiment", () => {
  it("loads the same PCA rows", () => {
    expect(pca.count).toBe(fx.pcaRowCount);
  });

  it("uses the source angle convention (ANGLE_MAP)", () => {
    fx.angles.indices.forEach((i, n) => {
      expect(Math.abs(angleDeg(pca.pc1[i], pca.pc2[i]) - fx.angles.angleDeg[n])).toBeLessThanOrEqual(1e-9);
    });
  });

  it("selects the same base and opposite points (create_base_and_opposite_points)", () => {
    for (const ref of fx.basePoints) {
      const index = createBasePointIndex(pca, ref.angle, sci);
      expect(index).toBe(ref.index);
      expect([pca.pc1[index], pca.pc2[index]]).toEqual(ref.base);
      const centers = initialCenters(pca, {
        ...experimentConfig,
        training: { ...experimentConfig.training, initialAngleA: ref.angle, initialAngleB: ref.angle + 180 },
      });
      expect(centers.centerA).toEqual(ref.base);
      expect(centers.centerB).toEqual(ref.opposite);
    }
  });

  it("rotates centers incrementally like rotate_vector", () => {
    for (const ref of fx.rotations) {
      let v = ref.start;
      for (let i = 0; i < ref.n; i += 1) v = rotateVector(v, ref.step);
      expectVecClose(v, ref.end);
    }
  });

  it("builds the same k-nearest clusters (collect_nearest_images)", () => {
    for (const ref of fx.clusters) {
      const cluster = collectNearest(ref.center, pca, ref.k);
      expect(cluster.map((c) => c.index)).toEqual(ref.indices);
    }
  });

  it("reproduces the training selection sequence (nearest not yet used, fixed centers)", () => {
    const centers = initialCenters(pca, experimentConfig);
    const used = new Set();
    fx.training.trials.forEach((ref) => {
      const stim = selectStimulus({ group: ref.group, centers, pca, excluded: used, clusterSize: 1 });
      expect(stim.imageIndex).toBe(ref.index);
      expect(Math.abs(stim.distanceToCenter - ref.distance)).toBeLessThanOrEqual(TOL);
      used.add(stim.imageIndex);
    });
  });

  it("reproduces the drift selection sequences (rotation after each response, no repeats in drift)", () => {
    for (const run of fx.drift) {
      const start = initialCenters(pca, experimentConfig);
      let centerA = start.centerA;
      let centerB = start.centerB;
      const used = new Set();
      run.trials.forEach((ref) => {
        expectVecClose(centerA, ref.centerA);
        expectVecClose(centerB, ref.centerB);
        const stim = selectStimulus({ group: ref.group, centers: { centerA, centerB }, pca, excluded: used, clusterSize: 1 });
        expect(stim.imageIndex).toBe(ref.index);
        used.add(stim.imageIndex);
        centerA = rotateVector(centerA, run.step);
        centerB = rotateVector(centerB, run.step);
      });
      expectVecClose(centerA, run.finalCenterA);
    }
  });

  it("has no near-ties that could make the selection order platform-dependent", () => {
    // For every drift/training choice, the runner-up must be clearly farther than the winner.
    const check = (center, excluded) => {
      const [first, second] = collectNearest(center, pca, 2, excluded);
      expect(second.distance - first.distance).toBeGreaterThan(1e-12);
    };
    const start = initialCenters(pca, experimentConfig);
    const usedTraining = new Set();
    for (const ref of fx.training.trials) {
      check(ref.group === "A" ? start.centerA : start.centerB, usedTraining);
      usedTraining.add(ref.index);
    }
    for (const run of fx.drift) {
      const used = new Set();
      for (const ref of run.trials) {
        check(ref.group === "A" ? ref.centerA : ref.centerB, used);
        used.add(ref.index);
      }
    }
  });
});
