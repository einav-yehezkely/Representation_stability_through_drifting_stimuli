// Interactive wrapper around the PCA plot: group filter,
// sample details with thumbnail, SVG/PNG export. Used by the development panel and the
// researcher pages only — never by the participant view.

import { h } from "../components/dom.js";
import { createPcaPlot } from "./pcaPlot.js";
import { exportPng, exportSvg } from "./plotExport.js";
import { driftCandidateIndices } from "./trajectoryMembership.js";

export function createPlotViewer({ pca, config, facesDirectory, colorBy = "group" }) {
  const plot = createPcaPlot({
    pca,
    trajectoryIndices: driftCandidateIndices(pca, config),
    colorBy,
  });

  const state = { group: "all", view: null };
  const select = (name, options) =>
    h(
      "select",
      { "data-filter": name, onchange: (e) => ((state[name] = e.target.value), render()) },
      options.map(([value, label]) => h("option", { value }, label)),
    );
  const count = h("span", { class: "plot-count" });
  const details = h("div", { class: "plot-details" }, "Click a sample to inspect it.");

  const fileBase = () => `${state.view?.sessionId ?? "session"}_pca_plot`;
  const controls = h(
    "div",
    { class: "plot-controls" },
    h("label", {}, "Sample group ", select("group", [["all", "All"], ["A", "A"], ["B", "B"]])),
    count,
    h("button", { type: "button", class: "secondary", onclick: () => exportSvg(plot.svg, `${fileBase()}.svg`) }, "Export SVG"),
    h("button", { type: "button", class: "secondary", onclick: () => exportPng(plot.svg, `${fileBase()}.png`) }, "Export PNG"),
  );

  function showDetails(r) {
    details.replaceChildren(
      h("img", { src: `${facesDirectory}${encodeURIComponent(r.sampleImageId)}`, alt: r.sampleImageId, width: 89, height: 109 }),
      h(
        "dl",
        {},
        [
          ["Trial", String(r.trialNumber)],
          ["Sample", r.sampleImageId],
          ["Sample group", r.sampleGroup],
          [(pca.axes ?? ["PC1", "PC2"]).join(", "), `${Number(r.samplePc1).toFixed(4)}, ${Number(r.samplePc2).toFixed(4)}`],
          ["Sample angle", `${Number(r.sampleAngle).toFixed(2)}°`],
          ["Centers (nominal)", `A ${r.currentAngleA}° · B ${r.currentAngleB}°`],
          ["Answer", `${r.response}, group ${r.chosenGroup} (${r.correct === true || r.correct === "true" ? "correct" : "incorrect"})`],
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
      filters: { group: state.group },
      onSelect: showDetails,
      title,
    });
    count.textContent = `${shown} samples shown`;
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
