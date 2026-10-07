// Trajectory membership for the researcher visualization (open question OQ2, default):
//
//   * driftCandidates: images that are nearest (cluster of size 1, no exclusion) to the A or B
//     center at any drift position of the effective configuration (training centers + the
//     incremental rotation used during drift).
//   * sourceRotationSequence: the source experiment's own trajectory sequence,
//     generate_rotation_sequence(base_point, num_steps=360, start_angle=0, rotation_range=360,
//     used_indices=set()) (classify_rotation_resnet50.py lines 760-804 and 1491-1500), offered as
//     an optional layer.

import { collectNearest } from "../scientific/clusterConstruction.js";
import { initialCenters, rotateVector } from "../scientific/trajectory.js";

export function driftCandidateIndices(pca, config) {
  const start = initialCenters(pca, config);
  let a = start.centerA;
  let b = start.centerB;
  const members = new Set();
  for (let k = 0; k < config.drift.numberOfTrials; k += 1) {
    members.add(collectNearest(a, pca, 1)[0].index);
    members.add(collectNearest(b, pca, 1)[0].index);
    a = rotateVector(a, config.drift.degreesPerTrial);
    b = rotateVector(b, config.drift.degreesPerTrial);
  }
  return [...members];
}

/** Port of generate_rotation_sequence (angle_deg = (start + range * i / steps) % 360). */
export function sourceRotationSequence(pca, basePoint, { numSteps = 360, startAngle = 0, rotationRange = 360 } = {}) {
  const used = new Set();
  const results = [];
  for (let i = 0; i < numSteps; i += 1) {
    const angle = (startAngle + (rotationRange * i) / numSteps) % 360;
    const rotated = rotateVector(basePoint, angle);
    const [{ index }] = collectNearest(rotated, pca, 1, used);
    used.add(index);
    results.push({ step: i, index });
  }
  return results;
}
