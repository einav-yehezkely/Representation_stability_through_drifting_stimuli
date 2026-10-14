// Trajectory membership for the researcher visualization (open question OQ2, default):
//
//   * driftCandidates: images that are nearest (cluster of size 1, no exclusion) to the A or B
//     center at any drift position of the effective configuration (training centers + the
//     incremental rotation used during drift).

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

