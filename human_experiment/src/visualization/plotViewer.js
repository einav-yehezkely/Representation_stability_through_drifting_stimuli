// Interactive wrapper around the PCA plot: phase/group filters,
// stimulus details with thumbnail, SVG/PNG export. Used by the development panel and the
// researcher review page only — never by the participant view.

import { h } from "../components/dom.js";
import { createPcaPlot } from "./pcaPlot.js";
import { exportPng, exportSvg } from "./plotExport.js";
import { driftCandidateIndices } from "./trajectoryMembership.js";

export function createPlotViewer({ pca, config, facesDirectory }) {
  const plot = createPcaPlot({
    pca,
    trajectoryIndices: driftCandidateIndices(pca, config),
  });

  const state = { phase: "all", group: "all", view: null };
  const select = (name, options) =>
    h(
      "select",
      { "data-filter": name, onchange: (e) => ((state[name] = e.target.value), render()) },
      options.map(([value, label]) => h("option", { value }, label)),
    );
  const count = h("span", { class: "plot-count" });
  const details = h("div", { class: "plot-details" }, "Click a presented stimulus to inspect it.");

  const fileBase = () => `${state.view?.sessionId ?? "session"}_pca_plot`;
  const controls = h(
    "div",
    { class: "plot-controls" },
    h("label", {}, "Phase ", select("phase", [["all", "All"], ["training", "Training"], ["drift", "Drift"]])),
    h("label", {}, "Group ", select("group", [["all", "All"], ["A", "A"], ["B", "B"]])),
    count,
    h("button", { type: "button", class: "secondary", onclick: () => exportSvg(plot.svg, `${fileBase()}.svg`) }, "Export SVG"),
    h("button", { type: "button", class: "secondary", onclick: () => exportPng(plot.svg, `${fileBase()}.png`) }, "Export PNG"),
  );

  function showDetails(r) {
    details.replaceChildren(
      h("img", { src: `${facesDirectory}${encodeURIComponent(r.imageId)}`, alt: r.imageId, width: 89, height: 109 }),
      h(
        "dl",
        {},
        [
          ["Trial", `${r.trialNumber} (${r.phase} #${r.phaseTrialNumber})`],
          ["Image", r.imageId],
          ["True group", r.trueGroup],
          [(pca.axes ?? ["PC1", "PC2"]).join(", "), `${Number(r.pc1).toFixed(4)}, ${Number(r.pc2).toFixed(4)}`],
          ["Stimulus angle", `${Number(r.stimulusAngle).toFixed(2)}°`],
          ["Centers (nominal)", `A ${r.currentAngleA}° · B ${r.currentAngleB}°`],
          ["Distance to center", Number(r.distanceToCenter).toFixed(4)],
          ["Response", `${r.response} (${r.correct === true || r.correct === "true" ? "correct" : "incorrect"})`],
        ].flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })]),
      ),
    );
  }

  function render() {
    if (!state.view) return;
    const { records, centers, current, title } = state.view;
    const shown = plot.update({
      records,
      centers,
      current,
      filters: { phase: state.phase, group: state.group },
      onSelect: showDetails,
      title,
    });
    count.textContent = `${shown} presented stimuli shown`;
  }

  const element = h("div", { class: "plot-viewer" }, controls, h("div", { class: "plot-svg" }, plot.svg), details);

  return {
    element,
    svg: plot.svg,
    /** view: { sessionId, records, centers, current, title } */
    setView(view) {
      state.view = view;
      render();
    },
  };
}
