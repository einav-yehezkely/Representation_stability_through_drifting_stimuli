// Renders engine state into the page and forwards participant actions to the engine.
// The UI holds no scientific or phase logic.

import { uiText } from "./content/uiText.js";
import { SCREENS } from "./experiment/experimentEngine.js";
import {
  completionScreen,
  consentScreen,
  errorScreen,
  instructionsScreen,
  loadingScreen,
  transitionScreen,
} from "./components/screens.js";

export function mountExperiment(root, engine, trialScreen, { developmentMode = false } = {}) {
  let current = null;
  let renderCount = 0;

  const show = (element, key) => {
    if (current === key && key === SCREENS.TRIAL) return;
    current = key;
    root.replaceChildren(element);
  };

  const errorMessage = (state) => {
    if (state.kind === "imageLoad") return uiText.errors.imageLoad;
    if (state.kind === "storage") return uiText.errors.storage;
    return state.message;
  };

  return engine.subscribe((state) => {
    switch (state.screen) {
      case SCREENS.CONSENT:
        show(
          consentScreen({
            onAccept: () => engine.acceptConsent(),
            onBypass: developmentMode ? () => engine.bypassConsent() : null,
          }),
          SCREENS.CONSENT,
        );
        break;
      case SCREENS.INSTRUCTIONS:
        show(instructionsScreen({ onStart: () => engine.startTraining() }), SCREENS.INSTRUCTIONS);
        break;
      case SCREENS.TRIAL:
        show(trialScreen.element, SCREENS.TRIAL);
        // Feedback text exists only in training states; drift states never carry it.
        trialScreen.setFeedback(state.status === "feedback" ? state.feedback : null);
        break;
      case SCREENS.TRANSITION:
        show(transitionScreen({ onContinue: () => engine.continueToDrift() }), SCREENS.TRANSITION);
        break;
      case SCREENS.COMPLETING:
        show(loadingScreen(), SCREENS.COMPLETING);
        break;
      case SCREENS.COMPLETE:
        show(completionScreen({ state, onDownload: () => engine.downloadAgain() }), `${SCREENS.COMPLETE}:${state.exported}`);
        break;
      case SCREENS.ERROR:
        show(
          errorScreen({
            message: errorMessage(state),
            detail: developmentMode || state.kind === "storage" ? state.message : null,
            onRetry: state.retry ? () => engine.retry() : null,
          }),
          `${SCREENS.ERROR}:${(renderCount += 1)}`, // always re-render errors
        );
        break;
      default:
        break;
    }
  });
}
