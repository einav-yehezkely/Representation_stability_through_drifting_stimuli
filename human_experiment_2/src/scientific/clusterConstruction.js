// Cluster construction, ported from collect_nearest_images
// (network_classification/classify_rotation_resnet50.py lines 321-357) with the exclusion
// mechanism of generate_rotation_sequence (lines 795-802: dists[idx] = inf for used indices).

import { norm2 } from "./trajectory.js";

/**
 * Indices of the k points nearest (2D Euclidean) to `center`, sorted by distance.
 * Indices in `excluded` are skipped (equivalent to the source's distance = inf).
 * Ties are broken by lower index; numpy's argpartition/argsort order on exact ties is
 * implementation-defined, and parity tests check that no ties occur in practice.
 *
 * Returns [{ index, distance }, ...] of length min(k, available).
 */
export function collectNearest(center, pca, k, excluded) {
  const [cx, cy] = center;
  if (k === 1) {
    let best = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < pca.count; i += 1) {
      if (excluded && excluded.has(i)) continue;
      const d = norm2(pca.pc1[i] - cx, pca.pc2[i] - cy);
      if (d < bestDistance) {
        bestDistance = d;
        best = i;
      }
    }
    return best === -1 ? [] : [{ index: best, distance: bestDistance }];
  }
  const candidates = [];
  for (let i = 0; i < pca.count; i += 1) {
    if (excluded && excluded.has(i)) continue;
    candidates.push({ index: i, distance: norm2(pca.pc1[i] - cx, pca.pc2[i] - cy) });
  }
  candidates.sort((a, b) => a.distance - b.distance || a.index - b.index);
  return candidates.slice(0, k);
}
