// Experiment settings the researcher can change on the admin page, without a new build:
// the initial center angles and the drift movement. They are stored on the server
// (save.php?settings=1 reads them, admin_api.php writes them) or, under the Vite dev server, in
// this browser's localStorage. Every new session applies them on top of experimentConfig; the
// resulting effectiveConfig is recorded in the session metadata as before.
//
// Center B is always opposite center A (decision D3: B = -A), so the two angles stay 180° apart.
// The number of drift trials defaults to one full rotation: 360 / degreesPerTrial.

import { findConfigProblems, resolveEffectiveConfig } from "./validateConfig.js";

export const SETTINGS_KEYS = ["training.initialAngleA", "training.initialAngleB", "drift.degreesPerTrial", "drift.numberOfTrials"];

export class SettingsError extends Error {
  constructor(message) {
    super(message);
    this.name = "SettingsError";
  }
}

export const normalizeAngle = (deg) => ((deg % 360) + 360) % 360;

/** The angle opposite `deg`, in [0, 360). */
export const oppositeAngle = (deg) => normalizeAngle(deg + 180);

/** Trials for one full rotation at `degreesPerTrial` (rounded when 360 is not a multiple). */
export function trialsForFullRotation(degreesPerTrial) {
  return Number.isFinite(degreesPerTrial) && degreesPerTrial > 0 ? Math.max(1, Math.round(360 / degreesPerTrial)) : null;
}

/** The settings values of a configuration. */
export function settingsFromConfig(config) {
  return Object.fromEntries(SETTINGS_KEYS.map((path) => [path, path.split(".").reduce((o, k) => o[k], config)]));
}

/** experimentConfig with the saved settings applied (frozen copy; null/undefined = defaults). */
export function applySettings(baseConfig, values) {
  if (!values) return resolveEffectiveConfig(baseConfig, {});
  return resolveEffectiveConfig(baseConfig, Object.fromEntries(SETTINGS_KEYS.filter((k) => k in values).map((k) => [k, values[k]])));
}

/** Problems with a settings object (empty when valid). */
export function findSettingsProblems(values, baseConfig, { pcaCount } = {}) {
  const problems = [];
  for (const key of SETTINGS_KEYS) {
    if (typeof values?.[key] !== "number" || !Number.isFinite(values[key])) problems.push(`${key} must be a number`);
  }
  if (problems.length) return problems;
  if (!(values["drift.degreesPerTrial"] > 0 && values["drift.degreesPerTrial"] <= 360)) problems.push("Degrees per trial must be greater than 0 and at most 360.");
  return [...problems, ...findConfigProblems(applySettings(baseConfig, values), { pcaCount })];
}

const localKey = (prefix) => `${prefix}:settings`;

/**
 * Load the saved settings: { values, updatedAt, source } with values = null when none are saved.
 * Throws SettingsError when they cannot be read (the experiment must not silently fall back).
 */
export async function loadExperimentSettings({ source, endpoint, fetch = globalThis.fetch?.bind(globalThis), localStorage = globalThis.localStorage, prefix = "hx" }) {
  if (source === "local") {
    let saved = null;
    try {
      saved = JSON.parse(localStorage?.getItem(localKey(prefix)) ?? "null");
    } catch {
      throw new SettingsError("The saved experiment settings in this browser are unreadable.");
    }
    return { values: saved?.values ?? null, updatedAt: saved?.updatedAt ?? null, source: "local" };
  }
  let reply;
  try {
    const response = await fetch(`${endpoint}?settings=1`, { cache: "no-store" });
    reply = await response.json();
    if (!response.ok || reply?.ok !== true) throw new Error(reply?.error ?? `HTTP ${response.status}`);
  } catch (error) {
    throw new SettingsError(`The experiment settings could not be loaded (${error.message}).`);
  }
  return { values: reply.settings?.values ?? null, updatedAt: reply.settings?.updatedAt ?? null, source: "server" };
}

/** Development (Vite dev server): settings saved in this browser. */
export function saveLocalSettings(values, { localStorage = globalThis.localStorage, prefix = "hx", now = () => new Date() } = {}) {
  const saved = { values, updatedAt: now().toISOString() };
  localStorage.setItem(localKey(prefix), JSON.stringify(saved));
  return saved;
}
