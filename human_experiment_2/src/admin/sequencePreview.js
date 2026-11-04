// Admin: the trials every session sees (sequencePlanner.planSession with the fixed seed of
// experimentConfig.randomization). The seed is shown read-only; it can only be changed in the
// configuration file. Answers never change the sequence.

import { h } from "../components/dom.js";
import { findConfigProblems } from "../config/validateConfig.js";
import { planSession } from "../experiment/sequencePlanner.js";
import { card } from "./format.js";
import { createImageGrid, imageDetails } from "./imageGrid.js";

function plannedItem(t) {
  return {
    trialNumber: t.trialNumber,
    sampleGroup: t.sampleGroup,
    sampleImageId: t.sample.imageId,
    optionImageIds: { A: t.options.A.imageId, B: t.options.B.imageId },
    leftGroup: t.leftGroup,
    planned: t,
  };
}

export function createSequencePreview({ getPca, config, facesDirectory }) {
  const seed = config.randomization.fixedSeed;
  const message = h("p", { class: "muted" });
  const details = h("div", { class: "image-details-host" }, h("p", { class: "muted", text: "Select a trial to see its details." }));
  const grid = createImageGrid({
    facesDirectory,
    onSelect: (item) => {
      const t = item.planned;
      details.replaceChildren(
        imageDetails(facesDirectory, item, [
          ["Trial", String(t.trialNumber)],
          ["Sample", `${t.sample.imageId} · group ${t.sampleGroup}${t.sampleSourceTrial ? ` · from trial ${t.sampleSourceTrial}` : ""}`],
          ["Option A", `${t.options.A.imageId} · ${t.options.A.stimulusAngle.toFixed(2)}°`],
          ["Option B", `${t.options.B.imageId} · ${t.options.B.stimulusAngle.toFixed(2)}°`],
          ["Left / right", `${t.leftGroup} / ${t.rightGroup}`],
          ["Centers (nominal)", `A ${t.snapshot.nominalAngleA}° · B ${t.snapshot.nominalAngleB}°`],
          ["Distance to center", `A ${t.options.A.distanceToCenter.toFixed(4)} · B ${t.options.B.distanceToCenter.toFixed(4)}`],
        ]),
      );
    },
  });

  function show() {
    const pca = getPca();
    if (!pca) return (message.textContent = "PCA data is still loading…");
    const problems = findConfigProblems(config, { pcaCount: pca.count });
    if (problems.length) return (message.textContent = `The configuration cannot be planned: ${problems.join("; ")}`);
    const plan = planSession({ pca, config, sessionSeed: seed });
    grid.setItems(plan.trials.map(plannedItem));
    message.textContent = `All ${plan.trials.length} trials.`;
  }

  const element = h(
    "div",
    { class: "view" },
    h("h1", { text: "Planned image order" }),
    h(
      "p",
      { class: "muted intro" },
      "Every session uses the same fixed seed, so every participant sees the same trials in this order. Image selection is deterministic: each option is the nearest unused image to its group's center, and the seed fixes which option of the previous trial becomes the next sample and on which side each group appears. Answers never change the sequence.",
    ),
    h("div", { class: "filters" }, h("span", { class: "field" }, "Seed ", h("strong", { class: "mono seed-value", text: String(seed) }), h("span", { class: "muted small", text: "(fixed in experimentConfig.js)" }))),
    message,
    card(null, h("span", { class: "legend" }, h("span", { class: "key key-a", text: "A" }), "Group A", h("span", { class: "key key-b", text: "B" }), "Group B", h("span", { class: "muted", text: "· each trial: sample, then the left and right faces" })), h("div", { class: "image-order" }, grid.element, details)),
  );

  return { element, show };
}
