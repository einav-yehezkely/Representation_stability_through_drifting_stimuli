// Development mode only: debug information, finish-early, partial CSV export and the
// live researcher PCA plot. Loaded with a dynamic import, never in participant mode.

import "../visualization/plot.css";
import "./dev.css";
import { h } from "../components/dom.js";
import { createPlotViewer } from "../visualization/plotViewer.js";

const fmt = (v, digits = 2) => (v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(digits).replace(/\.?0+$/, "") : String(v));

export function mountDevPanel(engine, { pca, config }) {
  const info = h("dl", { class: "dev-info" });
  const message = h("div", { class: "dev-message" });
  let viewer = null;
  const plotHost = h("div", { class: "dev-plot", hidden: true });

  const run = async (action) => {
    try {
      const result = await action();
      if (result?.exportName) message.textContent = result.exported ? `Downloaded ${result.exportName}` : `Download failed: ${result.exportError}`;
    } catch (error) {
      message.textContent = error.message;
    }
  };

  const finishButton = h("button", { type: "button", class: "secondary", onclick: () => run(() => engine.finishEarly()) }, "Finish early");
  const partialButton = h("button", { type: "button", class: "secondary", onclick: () => run(() => engine.exportPartial()) }, "Download partial CSV");
  const plotButton = h(
    "button",
    {
      type: "button",
      class: "secondary",
      onclick: () => {
        plotHost.hidden = !plotHost.hidden;
        if (!plotHost.hidden && !viewer) {
          viewer = createPlotViewer({ pca, config, facesDirectory: config.assets.facesDirectory });
          plotHost.append(viewer.element);
        }
        refresh();
      },
    },
    "PCA plot",
  );
  const reviewLink = h("a", { href: "review.html", target: "_blank", rel: "noopener" }, "Open review page");

  const panel = h(
    "aside",
    { class: "dev-panel", "data-dev": "panel" },
    h("div", { class: "dev-header" }, h("strong", { text: "DEVELOPMENT MODE" }), reviewLink),
    info,
    h("div", { class: "dev-actions" }, finishButton, partialButton, plotButton),
    message,
    plotHost,
  );
  document.body.classList.add("dev-mode");
  document.body.append(panel);

  function refresh() {
    const d = engine.debugInfo();
    const rows = [
      ["Screen", d.screen],
      ["Step", d.status],
      ["Trial #", d.trialNumber],
      ["Sample group", d.sampleGroup],
      ["Sample image", d.sampleImageId],
      ["Sample from trial", d.sampleSourceTrial],
      ["Sample angle", fmt(d.sampleAngle)],
      ["Option A", d.optionAImageId],
      ["Option B", d.optionBImageId],
      ["Left group", d.leftGroup],
      ["Center A (nominal / actual)", `${fmt(d.nominalAngleA)}° / ${fmt(d.actualAngleA)}°`],
      ["Center B (nominal / actual)", `${fmt(d.nominalAngleB)}° / ${fmt(d.actualAngleB)}°`],
      ["Answered trials", `${d.completedTrials} / ${config.drift.numberOfTrials}`],
      ["Seed", d.sessionSeed],
    ];
    info.replaceChildren(...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v ?? "—" })]));
    const started = engine.sessionStarted && !engine.finished;
    finishButton.disabled = !started;
    partialButton.disabled = !engine.sessionStarted || engine.records.length === 0;

    if (viewer && !plotHost.hidden) {
      const trial = engine.pending?.trial;
      const current = trial ? { trialNumber: trial.trialNumber, sampleGroup: trial.sampleGroup, samplePc1: trial.sample.pc1, samplePc2: trial.sample.pc2 } : null;
      viewer.setView({
        sessionId: engine.meta.sessionId,
        records: engine.records,
        centers: { centerA: d.centerA, centerB: d.centerB },
        current,
        title: `Session ${engine.meta.sessionId.slice(0, 8)} · A ${fmt(d.nominalAngleA)}° / B ${fmt(d.nominalAngleB)}°`,
      });
    }
  }

  engine.subscribe(refresh);
  return panel;
}
