// Composition root of the participant experiment:
// config -> storage adapter -> (recovery) -> PCA data -> engine -> UI.

import "./styles.css";
import { mountExperiment } from "./app.js";
import { errorScreen, loadingScreen, recoveryScreen } from "./components/screens.js";
import { TrialScreen } from "./components/trialScreen.js";
import { experimentConfig } from "./config/experimentConfig.js";
import { ConfigError, resolveEffectiveConfig, validateConfig } from "./config/validateConfig.js";
import { uiText } from "./content/uiText.js";
import { ExperimentEngine } from "./experiment/experimentEngine.js";
import { generateSessionSeed } from "./experiment/seededRandomization.js";
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
  if (!sessions.length) return true;
  return new Promise((resolve) => {
    const markAll = async () => {
      for (const s of sessions) await storage.markInterrupted(s.sessionId);
    };
    root.replaceChildren(
      recoveryScreen({
        sessions,
        onDownload: async () => {
          await markAll();
          for (const s of sessions) await storage.exportSession(s.sessionId, { incomplete: true });
          resolve(true);
        },
        onStartNew: async () => {
          await markAll();
          resolve(true);
        },
      }),
    );
  });
}

async function boot() {
  const root = document.getElementById("app");
  root.replaceChildren(loadingScreen());
  const developmentMode = isDevelopmentMode(window.location.search);

  let storage;
  try {
    storage = createStorage(experimentConfig);
  } catch (error) {
    fatal(root, uiText.errors.storage, error);
    return;
  }
  if (!(await handleRecovery(root, storage))) return;
  root.replaceChildren(loadingScreen());

  let pca;
  try {
    validateConfig(experimentConfig);
    pca = await loadPca(experimentConfig);
  } catch (error) {
    fatal(root, error instanceof ConfigError ? uiText.errors.config : uiText.errors.assets, error);
    return;
  }

  let overrides = {};
  let seed = null;
  let devModule = null;
  if (developmentMode) {
    const setup = await import("./dev/devSetup.js");
    ({ overrides, seed } = await setup.showDevSetup(root, experimentConfig, { pcaCount: pca.count }));
    devModule = await import("./dev/devPanel.js");
  }

  let config;
  try {
    config = validateConfig(resolveEffectiveConfig(experimentConfig, overrides), { pcaCount: pca.count });
  } catch (error) {
    fatal(root, uiText.errors.config, error);
    return;
  }

  const developmentOverrides = { ...overrides };
  if (seed !== null) developmentOverrides.sessionSeed = seed;

  const engine = new ExperimentEngine({
    config,
    developmentMode,
    developmentOverrides,
    pca,
    storage,
    presenter: null,
    sessionId: createSessionId(),
    sessionSeed: seed ?? generateSessionSeed(),
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
