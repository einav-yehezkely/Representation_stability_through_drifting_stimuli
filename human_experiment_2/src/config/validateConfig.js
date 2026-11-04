// Configuration validation and effective-configuration resolution.

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
  selection: ["nearestUnused"],
  algorithm: ["sfc32"],
  adapter: ["localCsv", "memory", "server"],
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
  const d = config.drift ?? {};
  const timing = config.timing ?? {};
  const s = config.scientific ?? {};
  const r = config.randomization ?? {};

  for (const key of ["initialAngleA", "initialAngleB"]) {
    if (typeof d[key] !== "number" || !Number.isFinite(d[key])) problems.push(`drift.${key} must be a finite number`);
  }
  if (Number.isFinite(d.initialAngleA) && Number.isFinite(d.initialAngleB)) {
    const separation = normalizeDeg(d.initialAngleB - d.initialAngleA);
    if (Math.abs(separation - 180) > 1e-9) {
      problems.push(`drift.initialAngleB must be 180° from initialAngleA; got a separation of ${separation}°`);
    }
  }
  if (!isPositiveInteger(d.numberOfTrials)) problems.push("drift.numberOfTrials must be a positive integer");
  if (typeof d.degreesPerTrial !== "number" || !Number.isFinite(d.degreesPerTrial)) {
    problems.push("drift.degreesPerTrial must be a finite number");
  }
  for (const key of ["sampleDurationMs", "blankDurationMs", "interTrialDurationMs"]) {
    if (!(Number.isFinite(timing[key]) && timing[key] >= 0)) problems.push(`timing.${key} must be a non-negative number`);
  }
  if (!isPositiveInteger(s.clusterSize)) problems.push("scientific.clusterSize must be a positive integer");
  if (Number.isFinite(pcaCount) && s.clusterSize > pcaCount) problems.push("scientific.clusterSize exceeds the number of PCA points");
  if (Number.isFinite(pcaCount)) {
    // Each trial selects two new images, and the first trial one more for its sample.
    if (isPositiveInteger(d.numberOfTrials) && 2 * d.numberOfTrials + 1 > pcaCount) {
      problems.push("drift.numberOfTrials needs more images than are available (two new images per trial, no repeats)");
    }
  }
  if (!(typeof s.targetRadius === "number" && s.targetRadius >= 0)) problems.push("scientific.targetRadius must be a non-negative number");
  if (!(typeof s.radiusErrorWeight === "number" && Number.isFinite(s.radiusErrorWeight))) {
    problems.push("scientific.radiusErrorWeight must be a finite number");
  }
  for (const key of ["angleConvention", "rotationDirection", "clusterDistance", "centerB", "selection"]) {
    if (!SUPPORTED[key].includes(s[key])) problems.push(`scientific.${key} "${s[key]}" is not supported`);
  }
  if (!SUPPORTED.algorithm.includes(r.algorithm)) problems.push(`randomization.algorithm "${r.algorithm}" is not supported`);
  if (r.seedPolicy !== "fixed") problems.push(`randomization.seedPolicy "${r.seedPolicy}" is not supported (only "fixed")`);
  if (!(Number.isInteger(r.fixedSeed) && r.fixedSeed >= 0 && r.fixedSeed <= 0xffffffff)) {
    problems.push("randomization.fixedSeed must be an integer between 0 and 4294967295");
  }
  if (config.ui?.preloadCount !== undefined && !(Number.isInteger(config.ui.preloadCount) && config.ui.preloadCount >= 0)) {
    problems.push("ui.preloadCount must be a non-negative integer");
  }
  if (!SUPPORTED.adapter.includes(config.storage?.adapter)) problems.push(`storage.adapter "${config.storage?.adapter}" is not supported`);
  if (config.storage?.adapter === "server") {
    if (typeof config.storage.serverEndpoint !== "string" || !config.storage.serverEndpoint) problems.push("storage.serverEndpoint must be a non-empty URL");
    if (!isPositiveInteger(config.storage.maxAttempts)) problems.push("storage.maxAttempts must be a positive integer");
    if (!(Number.isFinite(config.storage.requestTimeoutMs) && config.storage.requestTimeoutMs > 0)) problems.push("storage.requestTimeoutMs must be a positive number");
  }
  const completionUrl = config.completion?.prolificCompletionUrl ?? "";
  if (typeof completionUrl !== "string" || (completionUrl && !/^https:\/\//.test(completionUrl))) {
    problems.push("completion.prolificCompletionUrl must be empty or an https:// URL");
  }
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
