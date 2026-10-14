// Admin: the image order every session sees (sequencePlanner.planSession with the fixed seed of
// experimentConfig.randomization). The seed is shown read-only; it can only be changed in the
// configuration file. Training order is fixed by the seed (responses only decide how long
// training lasts); the drift order is fixed by the seed and independent of training.

import { h } from "../components/dom.js";
import { findConfigProblems } from "../config/validateConfig.js";
import { planSession } from "../experiment/sequencePlanner.js";
import { card } from "./format.js";
import { createImageGrid, imageDetails } from "./imageGrid.js";

function plannedItem(t, trialNumber) {
  return {
    trialNumber,
    phase: t.phase,
    phaseTrialNumber: t.phaseTrialNumber,
    group: t.group,
    imageId: t.stimulus.imageId,
    stimulusAngle: t.stimulus.stimulusAngle,
    angleA: t.snapshot.nominalAngleA,
    planned: t,
  };
}

export function createSequencePreview({ getPca, config, facesDirectory }) {
  const seed = config.randomization.fixedSeed;
  const trainingInput = h("input", { type: "number", min: 1, step: 1, "aria-label": "Training trials", value: String(config.training.minTrials * 2) });
  const message = h("p", { class: "muted" });
  const details = h("div", { class: "image-details-host" }, h("p", { class: "muted", text: "Select an image to see its details." }));
  const grid = createImageGrid({
    facesDirectory,
    onSelect: (item) => {
      const t = item.planned;
      details.replaceChildren(
        imageDetails(facesDirectory, item, [
          ["Position", `${t.phase} ${t.phaseTrialNumber}`],
          ["Image", t.stimulus.imageId],
          ["Group", t.group],
          ["Stimulus angle", `${t.stimulus.stimulusAngle.toFixed(2)}°`],
          ["Centers (nominal)", `A ${t.snapshot.nominalAngleA}° · B ${t.snapshot.nominalAngleB}°`],
          ["Distance to center", t.stimulus.distanceToCenter.toFixed(4)],
        ]),
      );
    },
  });

  function show() {
    const pca = getPca();
    const trainingTrials = Number(trainingInput.value);
    if (!pca) return (message.textContent = "PCA data is still loading…");
    if (!(Number.isInteger(trainingTrials) && trainingTrials > 0 && trainingTrials <= 2000)) return (message.textContent = "Training trials must be an integer between 1 and 2000.");
    const problems = findConfigProblems(config, { pcaCount: pca.count });
    if (problems.length) return (message.textContent = `The configuration cannot be planned: ${problems.join("; ")}`);
    const plan = planSession({ pca, config, sessionSeed: seed, trainingTrials });
    grid.setItems([...plan.training, ...plan.drift].map((t, i) => plannedItem(t, i + 1)));
    message.textContent = `The first ${plan.training.length} training images and all ${plan.drift.length} drift images.`;
  }

  const element = h(
    "div",
    { class: "view" },
    h("h1", { text: "Planned image order" }),
    h(
      "p",
      { class: "muted intro" },
      "Every session uses the same fixed seed, so every participant sees the faces in this order. Image selection is deterministic: the seed fixes the group order, and each face is the nearest unused image to its group's center. Answers only decide how many training images are shown; the drift images are the same however long training took.",
    ),
    h(
      "div",
      { class: "filters" },
      h("span", { class: "field" }, "Seed ", h("strong", { class: "mono seed-value", text: String(seed) }), h("span", { class: "muted small", text: "(fixed in experimentConfig.js)" })),
      h("label", { class: "field" }, "Training trials ", trainingInput),
      h("button", { type: "button", class: "small", onclick: show }, "Show order"),
    ),
    message,
    card(null, h("span", { class: "legend" }, h("span", { class: "key key-a", text: "A" }), "Group A", h("span", { class: "key key-b", text: "B" }), "Group B", h("span", { class: "muted", text: "· number = position in phase" })), h("div", { class: "image-order" }, grid.element, details)),
  );

  return { element, show };
}
