// Trajectory geometry, ported from network_classification/classify_rotation_resnet50.py
// (verbatim copy: human_experiment/reference_source/classify_rotation_resnet50.py).
// Arithmetic follows the numpy operation order so results match to floating-point precision.

const RAD_TO_DEG = 180 / Math.PI; // np.degrees / np.rad2deg multiply by 180/pi
const DEG_TO_RAD = Math.PI / 180; // np.deg2rad multiplies by pi/180

/** Normalize an angle in degrees to [0, 360). */
export function normalizeAngle(deg) {
  const a = deg % 360;
  const b = a < 0 ? a + 360 : a;
  return b >= 360 ? b - 360 : b;
}

/**
 * Angle of a PCA point in degrees, [0, 360).
 * Source ANGLE_MAP (lines 52-56): np.degrees(np.arctan2(y, x)) % 360.
 */
export function angleDeg(x, y) {
  return normalizeAngle(Math.atan2(y, x) * RAD_TO_DEG);
}

/** Euclidean norm as computed by np.linalg.norm for a 2-vector. */
export function norm2(x, y) {
  return Math.sqrt(x * x + y * y);
}

/**
 * Rotate a 2D vector counterclockwise by angleDegrees around the origin.
 * Source rotate_vector (lines 300-318): R = [[cos, -sin], [sin, cos]]; R @ v.
 */
export function rotateVector(v, angleDegrees) {
  const rad = angleDegrees * DEG_TO_RAD;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [c * v[0] + -s * v[1], s * v[0] + c * v[1]];
}

/**
 * Index of the data point used as the base (group A) center for targetAngle.
 * Source create_base_and_opposite_points (lines 255-297):
 *   angles_deg = (degrees(arctan2(y, x)) + 360) % 360
 *   delta = |angles_deg - target|; angle_error = min(delta, 360 - delta)
 *   combined_error = angle_error + |radius - target_radius| * 100
 *   base_idx = argmin(combined_error)   (first index on ties)
 */
export function createBasePointIndex(pca, targetAngle, { targetRadius, radiusErrorWeight }) {
  let bestIndex = -1;
  let bestError = Infinity;
  for (let i = 0; i < pca.count; i += 1) {
    const x = pca.pc1[i];
    const y = pca.pc2[i];
    const angle = (Math.atan2(y, x) * RAD_TO_DEG + 360) % 360;
    const delta = Math.abs(angle - targetAngle);
    const angleError = Math.min(delta, 360 - delta);
    const radiusError = Math.abs(norm2(x, y) - targetRadius);
    const combined = angleError + radiusError * radiusErrorWeight;
    if (combined < bestError) {
      bestError = combined;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Initial group centers. A is the source base point for initialAngleA; B = -A (decision D3,
 * source opposite_point = -base_point). Returns vectors plus provenance.
 */
export function initialCenters(pca, config) {
  const { initialAngleA, initialAngleB } = config.training;
  const baseIndex = createBasePointIndex(pca, initialAngleA, config.scientific);
  const a = [pca.pc1[baseIndex], pca.pc2[baseIndex]];
  const b = [-a[0], -a[1]];
  return {
    centerA: a,
    centerB: b,
    nominalAngleA: normalizeAngle(initialAngleA),
    nominalAngleB: normalizeAngle(initialAngleB),
    baseImageIndex: baseIndex,
    baseImageId: pca.imageIds[baseIndex],
  };
}

/** Nominal center angle for one-based drift trial k: initial + (k - 1) * step, in [0, 360). */
export function nominalDriftAngle(initialAngle, driftTrialNumber, degreesPerTrial) {
  return normalizeAngle(initialAngle + (driftTrialNumber - 1) * degreesPerTrial);
}
