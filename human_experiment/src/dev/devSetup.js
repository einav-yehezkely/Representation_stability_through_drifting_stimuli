// Development mode only: override form shown before the session starts (PRD §15).
// Every override is returned explicitly so it can be recorded in the session metadata.

import { h } from "../components/dom.js";
import { findConfigProblems, resolveEffectiveConfig } from "../config/validateConfig.js";

const FIELDS = [
  ["training.minTrials", "Minimum training trials", "integer"],
  ["training.accuracyWindow", "Accuracy window (last N trials)", "integer"],
  ["training.accuracyThreshold", "Accuracy threshold (0–1)", "number"],
  ["drift.numberOfTrials", "Drift trials", "integer"],
  ["drift.degreesPerTrial", "Degrees per drift trial", "number"],
];

export function showDevSetup(root, baseConfig, { pcaCount }) {
  return new Promise((resolve) => {
    const inputs = new Map();
    const problemsBox = h("ul", { class: "dev-problems" });
    const seedInput = h("input", { type: "text", inputmode: "numeric", placeholder: "random", "data-field": "seed" });

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
      const seedText = seedInput.value.trim();
      const seed = seedText === "" ? null : Number(seedText);
      return { overrides, seed };
    };

    const start = () => {
      const { overrides, seed } = collect();
      const problems = [];
      if (seed !== null && !(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff)) problems.push("Seed must be an integer between 0 and 4294967295.");
      problems.push(...findConfigProblems(resolveEffectiveConfig(baseConfig, overrides), { pcaCount }));
      if (problems.length) {
        problemsBox.replaceChildren(...problems.map((p) => h("li", { text: p })));
        return;
      }
      resolve({ overrides, seed });
    };

    root.replaceChildren(
      h(
        "section",
        { class: "panel dev-setup", "data-screen": "dev-setup" },
        h("h1", { text: "Development mode" }),
        h("p", {
          text: "These settings apply to this session only and are recorded in its metadata. Development sessions are marked as development data.",
        }),
        h("div", { class: "dev-grid" }, rows, h("label", { class: "dev-field" }, h("span", { text: "Session seed" }), seedInput)),
        problemsBox,
        h("div", { class: "actions" }, h("button", { type: "button", onclick: start }, "Start session")),
      ),
    );
  });
}
