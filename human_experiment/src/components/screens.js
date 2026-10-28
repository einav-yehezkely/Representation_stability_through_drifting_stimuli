// Participant-facing screens (English). Each function returns a DOM element.

import { consentContent } from "../content/consent.js";
import { instructionsContent } from "../content/instructions.js";
import { uiText } from "../content/uiText.js";
import { h } from "./dom.js";

// Static inline icons (no network request, no <img>: the trial screen's single <img> is the face).
const ICONS = {
  check: '<path d="M20 6 9 17l-5-5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
};

function icon(name, tone = "accent") {
  const badge = h("span", { class: `icon-badge icon-${tone}`, "aria-hidden": "true" });
  badge.innerHTML = `<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
  return badge;
}

export function consentScreen({ onAccept, onBypass }) {
  const continueButton = h("button", { type: "button", disabled: true, onclick: () => onAccept() }, consentContent.continueLabel);
  const label = h("label", { class: "consent-check", for: "consent-checkbox" });
  const checkbox = h("input", {
    type: "checkbox",
    id: "consent-checkbox",
    onchange: (event) => {
      continueButton.disabled = !event.target.checked;
      label.classList.toggle("checked", event.target.checked);
    },
  });
  label.append(checkbox, h("span", { text: consentContent.checkboxLabel }));
  return h(
    "section",
    { class: "panel", "data-screen": "consent" },
    h("p", { class: "eyebrow", text: "Study information" }),
    h("h1", { text: consentContent.title }),
    h(
      "div",
      { class: "consent-body", tabindex: "0" },
      consentContent.sections.map((section) => h("div", { class: "consent-section" }, h("h2", { text: section.heading }), h("p", { text: section.body }))),
    ),
    label,
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
    h("p", { class: "eyebrow", text: "Before you start" }),
    h("h1", { text: c.title }),
    c.paragraphs.map((text) => h("p", { text })),
    h(
      "ul",
      { class: "responses" },
      c.responseExplanations.map((item) => h("li", {}, h("strong", { class: "response-key", text: item.label }), h("span", { text: item.text }))),
    ),
    h(
      "div",
      { class: "note" },
      c.closingParagraphs.map((text) => h("p", { text })),
    ),
    h("div", { class: "actions" }, h("button", { type: "button", class: "large", onclick: () => onStart() }, c.startLabel)),
  );
}

export function transitionScreen({ onContinue }) {
  const t = uiText.transition;
  return h(
    "section",
    { class: "panel panel-centered", "data-screen": "transition" },
    icon("check", "success"),
    h("h1", { text: t.heading }),
    t.paragraphs.map((text) => h("p", { text })),
    h("div", { class: "actions" }, h("button", { type: "button", class: "large", onclick: () => onContinue() }, t.continueLabel)),
  );
}

// Participants are never offered their data: it is saved (server or browser storage) and the
// researcher exports it from the admin page.
export function completionScreen({ state, completionUrl }) {
  const c = uiText.completion;
  const complete = state.completionStatus === "complete";
  return h(
    "section",
    { class: "panel panel-centered", "data-screen": "complete" },
    icon(complete ? "check" : "info", complete ? "success" : "accent"),
    h("h1", { text: c.heading }),
    h("p", { class: "lead", text: complete ? c.completeMessage : c.incompleteMessage }),
    h("p", { text: c.savedMessage }),
    complete && completionUrl
      ? [h("p", { text: c.returnToProlificMessage }), h("div", { class: "actions" }, h("a", { class: "button large", href: completionUrl }, c.returnToProlificLabel))]
      : null,
  );
}

export function errorScreen({ message, detail, onRetry }) {
  return h(
    "section",
    { class: "panel panel-centered error-panel", "data-screen": "error", role: "alert" },
    icon("alert", "danger"),
    h("h1", { text: "Something went wrong" }),
    h("p", { text: message }),
    detail ? h("p", { class: "error-detail", text: detail }) : null,
    onRetry ? h("div", { class: "actions" }, h("button", { type: "button", onclick: () => onRetry() }, uiText.errors.retryLabel)) : null,
  );
}

export function loadingScreen() {
  return h(
    "section",
    { class: "loading", "data-screen": "loading", role: "status" },
    h("span", { class: "spinner", "aria-hidden": "true" }),
    h("p", { text: uiText.loading }),
  );
}
