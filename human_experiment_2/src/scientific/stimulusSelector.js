// Stimulus selection.
//
// The source presents every member of a k-nearest cluster to the network. Here the cluster size
// is 1, so each selected image is the single nearest image to its group's current center, skipping
// images already selected in the session. Selection is deterministic: no randomness is involved.

import { collectNearest } from "./clusterConstruction.js";
import { angleDeg } from "./trajectory.js";

export class StimulusExhaustedError extends Error {
  constructor() {
    super("No unused images remain for stimulus selection.");
    this.name = "StimulusExhaustedError";
  }
}

/**
 * @param {object} args
 * @param {"A"|"B"} args.group  group the stimulus is drawn for
 * @param {{centerA:number[], centerB:number[]}} args.centers  display-time snapshot
 * @param {object} args.pca      loaded PCA data
 * @param {Set<number>} args.excluded  indices already selected in this session
 * @param {number} args.clusterSize  configured cluster size (1)
 */
export function selectStimulus({ group, centers, pca, excluded, clusterSize }) {
  const center = group === "A" ? centers.centerA : centers.centerB;
  const cluster = collectNearest(center, pca, clusterSize, excluded);
  if (cluster.length === 0) throw new StimulusExhaustedError();
  // With clusterSize = 1 the cluster has exactly one member: the stimulus.
  const { index, distance } = cluster[0];
  const pc1 = pca.pc1[index];
  const pc2 = pca.pc2[index];
  return {
    imageIndex: index,
    imageId: pca.imageIds[index],
    pc1,
    pc2,
    stimulusAngle: angleDeg(pc1, pc2),
    distanceToCenter: distance,
  };
}
