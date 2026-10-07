// Development mode only (PRD §15): debug information, finish-early, partial CSV export and the
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
      ["Phase", d.phase],
      ["Trial #", d.trialNumber],
      ["Phase trial #", d.phaseTrialNumber],
      ["True group", d.group],
      ["Image", d.imageId],
      ["Stimulus angle", fmt(d.stimulusAngle)],
      ["Distance to center", fmt(d.distanceToCenter, 4)],
      ["Center A (nominal / actual)", `${fmt(d.nominalAngleA)}° / ${fmt(d.actualAngleA)}°`],
      ["Center B (nominal / actual)", `${fmt(d.nominalAngleB)}° / ${fmt(d.actualAngleB)}°`],
      ["Training trials", d.trainingCompleted],
      ["Window accuracy", fmt(d.windowAccuracy)],
      ["Criterion passed", String(d.criterionPassed)],
      ["Drift trials", `${d.driftCompleted} / ${config.drift.numberOfTrials}`],
      ["Seed", d.sessionSeed],
    ];
    info.replaceChildren(...rows.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v ?? "—" })]));
    const started = engine.sessionStarted && !engine.finished;
    finishButton.disabled = !started;
    partialButton.disabled = !engine.sessionStarted || engine.records.length === 0;

    if (viewer && !plotHost.hidden) {
      const pending = engine.pending;
      const current = pending
        ? {
            trialNumber: engine.records.length + 1,
            phase: pending.phase,
            trueGroup: pending.group,
            pc1: pending.stimulus.pc1,
            pc2: pending.stimulus.pc2,
          }
        : null;
      viewer.setView({
        sessionId: engine.meta.sessionId,
        records: engine.records,
        centers: { centerA: d.centerA, centerB: d.centerB },
        current,
        title: `Session ${engine.meta.sessionId.slice(0, 8)} · ${d.phase} · A ${fmt(d.nominalAngleA)}° / B ${fmt(d.nominalAngleB)}°`,
      });
    }
  }

  engine.subscribe(refresh);
  return panel;
}
