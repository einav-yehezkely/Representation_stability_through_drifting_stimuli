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
    savedMessage: "Your responses have been saved.",
    // The link itself is config.completion.prolificCompletionUrl.
    returnToProlificMessage: "Please click the button below to return to Prolific and register your submission.",
    returnToProlificLabel: "Return to Prolific",
  },
  errors: {
    imageLoad: "The picture could not be loaded. Please check your internet connection and try again.",
    storage: "Your answer could not be saved. Please check your internet connection and try again.",
    config: "The experiment could not be started because its configuration is invalid.",
    assets: "The experiment data could not be loaded. Please reload the page.",
    retryLabel: "Try again",
  },
};
