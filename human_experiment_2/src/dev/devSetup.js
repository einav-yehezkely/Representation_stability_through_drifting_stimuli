// Development mode only: override form shown before the session starts.
// Every override is returned explicitly so it can be recorded in the session metadata.

import { h } from "../components/dom.js";
import { findConfigProblems, resolveEffectiveConfig } from "../config/validateConfig.js";

const FIELDS = [
  ["drift.numberOfTrials", "Trials", "integer"],
  ["drift.degreesPerTrial", "Degrees per trial", "number"],
  ["timing.sampleDurationMs", "Sample duration (ms)", "number"],
  ["timing.blankDurationMs", "Blank duration (ms)", "number"],
  ["timing.interTrialDurationMs", "Empty screen after the answer (ms)", "number"],
];

export function showDevSetup(root, baseConfig, { pcaCount }) {
  return new Promise((resolve) => {
    const inputs = new Map();
    const problemsBox = h("ul", { class: "dev-problems" });

    const rows = FIELDS.map(([path, label, kind]) => {
      const [section, key] = path.split(".");
      const input = h("input", { type: "number", step: kind === "integer" ? "1" : "any", value: baseConfig[section][key], "data-field": path });
      inputs.set(path, { input, kind, base: baseConfig[section][key] });
      return h("label", { class: "dev-field" }, h("span", { text: label }), input);
    });

    const collect = () => {
      const overrides = {};
      for (const [path, { input, kind, base }] of inputs) {
        const value = kind === "integer" ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
        if (value !== base) overrides[path] = value;
      }
      return { overrides };
    };

    const start = () => {
      const { overrides } = collect();
      const problems = findConfigProblems(resolveEffectiveConfig(baseConfig, overrides), { pcaCount });
      if (problems.length) {
        problemsBox.replaceChildren(...problems.map((p) => h("li", { text: p })));
        return;
      }
      resolve({ overrides });
    };

    root.replaceChildren(
      h(
        "section",
        { class: "panel dev-setup", "data-screen": "dev-setup" },
        h("h1", { text: "Development mode" }),
        h("p", {
          text: "These settings apply to this session only and are recorded in its metadata. Development sessions are marked as development data.",
        }),
        h("p", { text: `Session seed: ${baseConfig.randomization.fixedSeed} (fixed for every session in experimentConfig.js).` }),
        h("div", { class: "dev-grid" }, rows),
        problemsBox,
        h("div", { class: "actions" }, h("button", { type: "button", onclick: start }, "Start session")),
      ),
    );
  });
}
