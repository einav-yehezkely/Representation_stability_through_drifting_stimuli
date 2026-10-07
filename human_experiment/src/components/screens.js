// Participant-facing screens (English). Each function returns a DOM element.

import { consentContent } from "../content/consent.js";
import { instructionsContent } from "../content/instructions.js";
import { uiText } from "../content/uiText.js";
import { h } from "./dom.js";

export function consentScreen({ onAccept, onBypass }) {
  const continueButton = h("button", { type: "button", disabled: true, onclick: () => onAccept() }, consentContent.continueLabel);
  const checkbox = h("input", {
    type: "checkbox",
    id: "consent-checkbox",
    onchange: (event) => {
      continueButton.disabled = !event.target.checked;
    },
  });
  return h(
    "section",
    { class: "panel", "data-screen": "consent" },
    h("h1", { text: consentContent.title }),
    h(
      "div",
      { class: "consent-body" },
      consentContent.sections.map((section) => [h("h2", { text: section.heading }), h("p", { text: section.body })]),
    ),
    h("label", { class: "consent-check", for: "consent-checkbox" }, checkbox, h("span", { text: consentContent.checkboxLabel })),
    h(
      "div",
      { class: "actions" },
      continueButton,
      onBypass ? h("button", { type: "button", class: "secondary", "data-dev": "bypass-consent", onclick: () => onBypass() }, "Bypass consent (development)") : null,
    ),
  );
}

export function instructionsScreen({ onStart }) {
  const c = instructionsContent;
  return h(
    "section",
    { class: "panel", "data-screen": "instructions" },
    h("h1", { text: c.title }),
    c.paragraphs.map((text) => h("p", { text })),
    h(
      "ul",
      { class: "responses" },
      c.responseExplanations.map((item) => h("li", {}, h("strong", { text: item.label }), ` — ${item.text}`)),
    ),
    c.closingParagraphs.map((text) => h("p", { text })),
    h("div", { class: "actions" }, h("button", { type: "button", onclick: () => onStart() }, c.startLabel)),
  );
}

export function transitionScreen({ onContinue }) {
  const t = uiText.transition;
  return h(
    "section",
    { class: "panel", "data-screen": "transition" },
    h("p", {}, h("strong", { text: t.heading })),
    t.paragraphs.map((text) => h("p", { text })),
    h("div", { class: "actions" }, h("button", { type: "button", onclick: () => onContinue() }, t.continueLabel)),
  );
}

export function completionScreen({ state, onDownload }) {
  const c = uiText.completion;
  const complete = state.completionStatus === "complete";
  return h(
    "section",
    { class: "panel", "data-screen": "complete" },
    h("h1", { text: c.heading }),
    h("p", { text: complete ? c.completeMessage : c.incompleteMessage }),
    h("p", { text: state.exported ? c.downloadedMessage.replace("{file}", state.exportName) : c.notDownloadedMessage }),
    h("div", { class: "actions" }, h("button", { type: "button", onclick: () => onDownload() }, c.downloadLabel)),
  );
}

export function recoveryScreen({ sessions, onDownload, onStartNew }) {
  const r = uiText.recovery;
  return h(
    "section",
    { class: "panel", "data-screen": "recovery" },
    h("h1", { text: r.heading }),
    h("p", { text: r.body }),
    h(
      "div",
      { class: "actions" },
      h("button", { type: "button", onclick: () => onDownload(sessions) }, r.downloadLabel),
      h("button", { type: "button", class: "secondary", onclick: () => onStartNew(sessions) }, r.startNewLabel),
    ),
  );
}

export function errorScreen({ message, detail, onRetry }) {
  return h(
    "section",
    { class: "panel error-panel", "data-screen": "error", role: "alert" },
    h("h1", { text: "Something went wrong" }),
    h("p", { text: message }),
    detail ? h("p", { class: "error-detail", text: detail }) : null,
    onRetry ? h("div", { class: "actions" }, h("button", { type: "button", onclick: () => onRetry() }, uiText.errors.retryLabel)) : null,
  );
}

export function loadingScreen() {
  return h("section", { class: "panel", "data-screen": "loading" }, h("p", { text: uiText.loading }));
}
