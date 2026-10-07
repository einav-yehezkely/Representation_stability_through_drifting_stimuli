// Configuration validation and effective-configuration resolution (PRD §10).

export class ConfigError extends Error {
  constructor(problems) {
    super(`Invalid experiment configuration:\n- ${problems.join("\n- ")}`);
    this.name = "ConfigError";
    this.problems = problems;
  }
}

const SUPPORTED = {
  angleConvention: ["atan2(PC2,PC1) in degrees, mod 360"],
  rotationDirection: ["counterclockwise"],
  clusterDistance: ["euclidean2D"],
  centerB: ["negateA"],
  trainingSelection: ["nearestUnused"],
  driftSelection: ["nearestUnused"],
  exclusionScope: ["perPhase"],
  algorithm: ["sfc32"],
  adapter: ["localCsv", "memory"],
};

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function normalizeDeg(value) {
  return ((value % 360) + 360) % 360;
}

/**
 * Validate a configuration object. Returns the list of problems (empty when valid).
 * `pcaCount` (optional) enables checks that depend on the loaded PCA data.
 */
export function findConfigProblems(config, { pcaCount } = {}) {
  const problems = [];
  const t = config.training ?? {};
  const d = config.drift ?? {};
  const s = config.scientific ?? {};
  const r = config.randomization ?? {};

  for (const key of ["initialAngleA", "initialAngleB"]) {
    if (typeof t[key] !== "number" || !Number.isFinite(t[key])) problems.push(`training.${key} must be a finite number`);
  }
  if (Number.isFinite(t.initialAngleA) && Number.isFinite(t.initialAngleB)) {
    const separation = normalizeDeg(t.initialAngleB - t.initialAngleA);
    if (Math.abs(separation - 180) > 1e-9) {
      problems.push(`training.initialAngleB must be 180° from initialAngleA (decision D3); got a separation of ${separation}°`);
    }
  }
  for (const key of ["minTrials", "accuracyWindow"]) {
    if (!isPositiveInteger(t[key])) problems.push(`training.${key} must be a positive integer`);
  }
  if (typeof t.accuracyThreshold !== "number" || !(t.accuracyThreshold >= 0 && t.accuracyThreshold <= 1)) {
    problems.push("training.accuracyThreshold must be a number in [0, 1]");
  }
  if (!isPositiveInteger(d.numberOfTrials)) problems.push("drift.numberOfTrials must be a positive integer");
  if (typeof d.degreesPerTrial !== "number" || !Number.isFinite(d.degreesPerTrial)) {
    problems.push("drift.degreesPerTrial must be a finite number");
  }
  if (!isPositiveInteger(s.clusterSize)) problems.push("scientific.clusterSize must be a positive integer");
  if (Number.isFinite(pcaCount) && s.clusterSize > pcaCount) problems.push("scientific.clusterSize exceeds the number of PCA points");
  if (Number.isFinite(pcaCount)) {
    // nearest-unused selection needs enough distinct images in each phase.
    if (isPositiveInteger(d.numberOfTrials) && d.numberOfTrials > pcaCount) {
      problems.push("drift.numberOfTrials exceeds the number of available images (no repeats in drift, decision D5)");
    }
  }
  if (!(typeof s.targetRadius === "number" && s.targetRadius >= 0)) problems.push("scientific.targetRadius must be a non-negative number");
  if (!(typeof s.radiusErrorWeight === "number" && Number.isFinite(s.radiusErrorWeight))) {
    problems.push("scientific.radiusErrorWeight must be a finite number");
  }
  for (const key of ["angleConvention", "rotationDirection", "clusterDistance", "centerB", "trainingSelection", "driftSelection", "exclusionScope"]) {
    if (!SUPPORTED[key].includes(s[key])) problems.push(`scientific.${key} "${s[key]}" is not supported`);
  }
  if (!SUPPORTED.algorithm.includes(r.algorithm)) problems.push(`randomization.algorithm "${r.algorithm}" is not supported`);
  if (!isPositiveInteger(r.trainingBlockSize) || r.trainingBlockSize % 2 !== 0) {
    problems.push("randomization.trainingBlockSize must be a positive even integer");
  }
  if (!(Number.isFinite(config.ui?.feedbackDurationMs) && config.ui.feedbackDurationMs >= 0)) {
    problems.push("ui.feedbackDurationMs must be a non-negative number");
  }
  if (!SUPPORTED.adapter.includes(config.storage?.adapter)) problems.push(`storage.adapter "${config.storage?.adapter}" is not supported`);
  for (const key of ["facesDirectory", "pcaDataPath", "manifestPath"]) {
    if (typeof config.assets?.[key] !== "string" || !config.assets[key]) problems.push(`assets.${key} must be a non-empty path`);
  }
  return problems;
}

export function validateConfig(config, options) {
  const problems = findConfigProblems(config, options);
  if (problems.length) throw new ConfigError(problems);
  return config;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
}

/**
 * Apply development overrides (flat "section.key" -> value map) to the base configuration.
 * Returns a frozen deep copy; the base configuration is never mutated.
 */
export function resolveEffectiveConfig(baseConfig, overrides = {}) {
  const effective = deepClone(baseConfig);
  for (const [path, value] of Object.entries(overrides)) {
    const [section, key] = path.split(".");
    if (!effective[section] || !(key in effective[section])) {
      throw new ConfigError([`unknown override "${path}"`]);
    }
    effective[section][key] = value;
  }
  return deepFreeze(effective);
}
