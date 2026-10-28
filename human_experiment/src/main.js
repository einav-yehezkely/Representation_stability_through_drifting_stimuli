// Composition root of the participant experiment:
// config -> storage adapter -> (recovery) -> PCA data -> engine -> UI.

import "./styles.css";
import { mountExperiment } from "./app.js";
import { errorScreen, loadingScreen } from "./components/screens.js";
import { TrialScreen } from "./components/trialScreen.js";
import { experimentConfig } from "./config/experimentConfig.js";
import { applySettings, loadExperimentSettings } from "./config/experimentSettings.js";
import { ConfigError, resolveEffectiveConfig, validateConfig } from "./config/validateConfig.js";
import { uiText } from "./content/uiText.js";
import { ExperimentEngine } from "./experiment/experimentEngine.js";
import { createSessionId, resolveParticipantId } from "./experiment/session.js";
import { loadPca } from "./scientific/pcaLoader.js";
import { createStorage } from "./services/dataStorage/index.js";

/**
 * Development mode is off by default. It is available only when explicitly enabled in the
 * central configuration, or when running the Vite dev server (`npm run dev`) with ?dev=1.
 * A production build ignores ?dev=1 unless config.development.enabled is true.
 */
function isDevelopmentMode(search) {
  if (experimentConfig.development.enabled) return true;
  return Boolean(import.meta.env.DEV) && new URLSearchParams(search).get("dev") === "1";
}

function fatal(root, message, error) {
  root.replaceChildren(errorScreen({ message, detail: error?.message ?? null }));
}

async function handleRecovery(root, storage) {
  let sessions;
  try {
    sessions = (await storage.listSessions()).filter((s) => s.meta?.completionStatus === "in_progress");
  } catch (error) {
    fatal(root, uiText.errors.storage, error);
    return false;
  }
  // The interrupted session keeps its saved data (server or this browser) and is marked
  // interrupted; a new session starts. Participants are never offered a download of their data.
  for (const s of sessions) await storage.markInterrupted(s.sessionId);
  return true;
}

async function boot() {
  const root = document.getElementById("app");
  root.replaceChildren(loadingScreen());
  const developmentMode = isDevelopmentMode(window.location.search);
  // The Vite dev server cannot run save.php, so it uses the development adapter.
  const adapter = import.meta.env.DEV ? experimentConfig.storage.developmentAdapter : experimentConfig.storage.adapter;
  const settle = (promise) => promise.then((value) => ({ value }), (error) => ({ error }));
  // Start downloading the PCA data and the admin-page settings right away, in parallel with the
  // storage/recovery step. Settings come from the server, or from this browser under the dev server.
  const pcaRequest = settle(loadPca(experimentConfig));
  const settingsRequest = settle(
    loadExperimentSettings({
      source: adapter === "server" ? "server" : "local",
      endpoint: experimentConfig.storage.serverEndpoint,
      prefix: experimentConfig.storage.localStoragePrefix,
    }),
  );

  let storage;
  try {
    storage = createStorage({ ...experimentConfig, storage: { ...experimentConfig.storage, adapter } });
  } catch (error) {
    fatal(root, uiText.errors.storage, error);
    return;
  }
  if (!(await handleRecovery(root, storage))) return;
  root.replaceChildren(loadingScreen());

  let pca;
  let settings;
  let baseConfig; // experimentConfig with the settings saved on the admin page
  try {
    validateConfig(experimentConfig);
    const loaded = await pcaRequest;
    if (loaded.error) throw loaded.error;
    pca = loaded.value;
    const loadedSettings = await settingsRequest;
    if (loadedSettings.error) throw loadedSettings.error;
    settings = loadedSettings.value;
    baseConfig = applySettings(experimentConfig, settings.values);
  } catch (error) {
    fatal(root, error instanceof ConfigError ? uiText.errors.config : uiText.errors.assets, error);
    return;
  }

  let overrides = {};
  let devModule = null;
  if (developmentMode) {
    const setup = await import("./dev/devSetup.js");
    ({ overrides } = await setup.showDevSetup(root, baseConfig, { pcaCount: pca.count }));
    devModule = await import("./dev/devPanel.js");
  }

  let config;
  try {
    config = validateConfig(resolveEffectiveConfig(baseConfig, overrides), { pcaCount: pca.count });
  } catch (error) {
    fatal(root, uiText.errors.config, error);
    return;
  }

  const engine = new ExperimentEngine({
    config,
    developmentMode,
    developmentOverrides: { ...overrides },
    experimentSettings: settings,
    pca,
    storage,
    presenter: null,
    sessionId: createSessionId(),
    sessionSeed: config.randomization.fixedSeed, // the same for every session
    identity: resolveParticipantId(window.location.search),
  });
  const trialScreen = new TrialScreen({
    scale: config.ui.imageDisplayScale,
    onResponse: (response, time) => engine.respond(response, time),
  });
  engine.presenter = trialScreen;

  mountExperiment(root, engine, trialScreen, { developmentMode });
  if (devModule) devModule.mountDevPanel(engine, { pca, config });
}

boot();
