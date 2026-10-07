// RESEARCHER-EDITABLE interface text (English).

export const uiText = {
  loading: "Loading…",
  transition: {
    // Exact wording required by the PRD (§4). Announces that feedback stops; never announces
    // category movement.
    heading: "The first part of the experiment is complete.",
    paragraphs: [
      "In the next part, you will continue performing the same task, but you will no longer receive feedback about your answers.",
      "Please continue to classify each face as accurately as possible.",
    ],
    continueLabel: "Continue",
  },
  completion: {
    heading: "Thank you!",
    completeMessage: "You have completed the experiment.",
    incompleteMessage: "The session has ended before completion and was saved as incomplete.",
    downloadedMessage: "A data file named {file} has been downloaded to this computer.",
    notDownloadedMessage: "The data file could not be downloaded automatically. Please use the button below.",
    downloadLabel: "Download data",
    // [RESEARCHER: completion code / redirect instructions for Prolific go here once a
    // server-backed storage adapter exists.]
  },
  recovery: {
    heading: "A previous session was interrupted",
    body: "A session in this browser ended before it was completed. Its saved responses are kept. You can download them, or start a new session.",
    downloadLabel: "Download its data",
    startNewLabel: "Start a new session",
  },
  errors: {
    imageLoad: "The picture could not be loaded. Please check your internet connection and try again.",
    storage: "Your answer could not be saved in this browser. Please try again.",
    config: "The experiment could not be started because its configuration is invalid.",
    assets: "The experiment data could not be loaded. Please reload the page.",
    retryLabel: "Try again",
  },
};
