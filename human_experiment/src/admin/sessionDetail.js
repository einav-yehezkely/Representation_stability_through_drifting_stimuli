// Admin: one session. Summary, accuracy over trials, the images in the order they were shown,
// the PCA stimulus plot and every trial record.

import { h } from "../components/dom.js";
import { browserDownload } from "../services/dataStorage/browserDownload.js";
import { createPlotViewer } from "../visualization/plotViewer.js";
import { createAccuracyChart } from "./accuracyChart.js";
import { card, formatDateTime, formatDuration, formatMs, median, percent, sessionCsv, statTile, statusBadge } from "./format.js";
import { createImageGrid, imageDetails } from "./imageGrid.js";

const fixed = (v, digits) => (v === null || v === undefined || v === "" ? "—" : Number(v).toFixed(digits));

function trialItem(r) {
  return {
    trialNumber: r.trialNumber,
    phase: r.phase,
    phaseTrialNumber: r.phaseTrialNumber,
    group: r.trueGroup,
    imageId: r.imageId,
    stimulusAngle: r.stimulusAngle,
    angleA: r.currentAngleA,
    response: r.response,
    correct: r.correct,
    record: r,
  };
}

function trialTable(trials) {
  const columns = [
    ["#", (r) => r.trialNumber, "num"],
    ["Phase", (r) => `${r.phase} ${r.phaseTrialNumber}`],
    ["Image", (r) => r.imageId, "mono"],
    ["Group", (r) => r.trueGroup],
    ["Answer", (r) => r.response],
    ["Correct", (r) => (r.correct ? "✓" : "✗")],
    ["RT (ms)", (r) => Math.round(r.responseTimeMs), "num"],
    ["Stimulus angle", (r) => `${fixed(r.stimulusAngle, 1)}°`, "num"],
    ["Center A / B", (r) => `${r.currentAngleA}° / ${r.currentAngleB}°`, "num"],
    ["Distance", (r) => fixed(r.distanceToCenter, 4), "num"],
    ["Window acc.", (r) => (r.windowAccuracy === null || r.windowAccuracy === undefined ? "" : `${Math.round(r.windowAccuracy * 100)}%`), "num"],
    ["Time", (r) => (r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : ""), "num"],
  ];
  return h(
    "div",
    { class: "table-scroll tall" },
    h(
      "table",
      { class: "data-table" },
      h("thead", {}, h("tr", {}, columns.map(([title]) => h("th", { text: title })))),
      h(
        "tbody",
        {},
        trials.map((r) =>
          h(
            "tr",
            { class: r.correct ? "" : "row-incorrect" },
            columns.map(([, value, cls]) => h("td", { class: cls ?? null, text: value(r) ?? "" })),
          ),
        ),
      ),
    ),
  );
}

export async function renderSessionDetail({ source, sessionId, pca, config, facesDirectory, onBack, onRefresh }) {
  const { meta, trials, status, lastActivity, warnings } = await source.getSession(sessionId);
  const effectiveConfig = meta.effectiveConfig ?? config;
  const training = trials.filter((t) => t.phase === "training");
  const drift = trials.filter((t) => t.phase === "drift");
  const criterionAt = training.find((t) => t.criterionPassed === true)?.trialNumber ?? null;
  const plannedDrift = effectiveConfig.drift?.numberOfTrials ?? null;

  const csv = () => {
    const { name, text } = sessionCsv(sessionId, meta, trials);
    browserDownload(name, text);
  };

  // Image order: thumbnails with filters and a details pane.
  const details = h("div", { class: "image-details-host" }, h("p", { class: "muted", text: "Select an image to see its details." }));
  const grid = createImageGrid({
    facesDirectory,
    onSelect: (item) => {
      const r = item.record;
      details.replaceChildren(
        imageDetails(facesDirectory, item, [
          ["Trial", `${r.trialNumber} (${r.phase} ${r.phaseTrialNumber})`],
          ["Image", r.imageId],
          ["True group", r.trueGroup],
          ["Answer", `${r.response} · ${r.correct ? "correct" : "incorrect"}`],
          ["Response time", formatMs(r.responseTimeMs)],
          ["PC1, PC2", `${fixed(r.pc1, 4)}, ${fixed(r.pc2, 4)}`],
          ["Stimulus angle", `${fixed(r.stimulusAngle, 2)}°`],
          ["Centers (nominal)", `A ${r.currentAngleA}° · B ${r.currentAngleB}°`],
          ["Distance to center", fixed(r.distanceToCenter, 4)],
          ["Feedback shown", r.feedbackShown ? "yes" : "no"],
        ]),
      );
    },
  });
  const filters = { phase: "all", group: "all", outcome: "all" };
  const applyFilters = () =>
    grid.setItems(
      trials
        .filter(
          (r) =>
            (filters.phase === "all" || r.phase === filters.phase) &&
            (filters.group === "all" || r.trueGroup === filters.group) &&
            (filters.outcome === "all" || (filters.outcome === "incorrect") !== r.correct),
        )
        .map(trialItem),
      { empty: trials.length ? "No images match the filters." : "No answered trials yet." },
    );
  const filterSelect = (name, options) =>
    h(
      "select",
      { "aria-label": name, onchange: (e) => ((filters[name] = e.target.value), applyFilters()) },
      options.map(([value, label]) => h("option", { value }, label)),
    );
  applyFilters();

  // PCA plot of the presented stimuli (same viewer as the development panel).
  const plotHost = h("div", {});
  if (pca && trials.length) {
    const viewer = createPlotViewer({ pca, config: effectiveConfig, facesDirectory });
    const last = trials.at(-1);
    viewer.setView({
      sessionId,
      records: trials,
      centers: { centerA: [last.centerA_pc1, last.centerA_pc2], centerB: [last.centerB_pc1, last.centerB_pc2] },
      current: last,
      title: `Session ${sessionId.slice(0, 8)} · trial ${last.trialNumber} (${last.phase}) · A ${last.currentAngleA}° / B ${last.currentAngleB}°`,
    });
    plotHost.append(viewer.element);
  } else {
    plotHost.append(h("p", { class: "muted", text: pca ? "No answered trials yet." : "PCA data is still loading…" }));
  }

  const end = meta.sessionEndTime ?? lastActivity;
  return h(
    "div",
    { class: "view" },
    h("button", { type: "button", class: "link", onclick: onBack }, "← All sessions"),
    h(
      "header",
      { class: "detail-header" },
      h(
        "div",
        {},
        h("h1", {}, `Session ${sessionId.slice(0, 8)} `, statusBadge(status), meta.developmentMode ? h("span", { class: "tag", text: "dev" }) : null),
        h(
          "p",
          { class: "muted" },
          `Participant `,
          h("span", { class: "mono", text: meta.participantId ?? "—" }),
          ` (${meta.participantIdSource ?? "?"}) · started ${formatDateTime(meta.sessionStartTime)} · ${formatDuration(meta.sessionStartTime, end)}${meta.sessionEndTime ? "" : " so far"} · seed `,
          h("span", { class: "mono", text: String(meta.sessionSeed) }),
          meta.consentBypassed ? " · consent bypassed (dev)" : meta.consentGiven ? " · consent given" : "",
        ),
        h(
          "p",
          { class: "muted small" },
          `Settings: center A ${effectiveConfig.training?.initialAngleA}° / B ${effectiveConfig.training?.initialAngleB}° · ${effectiveConfig.drift?.degreesPerTrial}° per drift trial × ${plannedDrift} trials`,
          meta.experimentSettings?.updatedAt ? ` (saved ${formatDateTime(meta.experimentSettings.updatedAt)})` : meta.experimentSettings ? " (defaults)" : "",
        ),
        h("p", { class: "muted small mono", text: sessionId }),
      ),
      h(
        "div",
        { class: "detail-actions" },
        status === "in_progress" ? h("button", { type: "button", class: "secondary small", onclick: onRefresh }, "Refresh") : null,
        h("button", { type: "button", class: "small", onclick: csv, disabled: trials.length === 0 }, "Download CSV"),
      ),
    ),
    warnings?.length ? h("p", { class: "warning", text: `File warnings: ${warnings.join("; ")}` }) : null,
    h(
      "div",
      { class: "stats" },
      statTile("Training trials", String(training.length), criterionAt ? `criterion met at trial ${criterionAt}` : "criterion not met"),
      statTile("Training accuracy", percent(training.filter((t) => t.correct).length, training.length), "all training trials"),
      statTile("Drift trials", `${drift.length} / ${plannedDrift ?? "?"}`),
      statTile("Drift accuracy", percent(drift.filter((t) => t.correct).length, drift.length), "no feedback"),
      statTile("Median RT", formatMs(median(trials.map((t) => Number(t.responseTimeMs)))), "image onset → answer"),
    ),
    card(`Rolling accuracy (last ${effectiveConfig.training?.accuracyWindow ?? 20} trials)`, createAccuracyChart({ records: trials, window: effectiveConfig.training?.accuracyWindow ?? 20 }).element),
    card(
      "Image order",
      h(
        "div",
        { class: "filters" },
        filterSelect("phase", [["all", "Both phases"], ["training", "Training"], ["drift", "Drift"]]),
        filterSelect("group", [["all", "Both groups"], ["A", "Group A"], ["B", "Group B"]]),
        filterSelect("outcome", [["all", "All answers"], ["correct", "Correct only"], ["incorrect", "Incorrect only"]]),
        h("span", { class: "legend" }, h("span", { class: "key key-a", text: "A" }), "Group A", h("span", { class: "key key-b", text: "B" }), "Group B", h("span", { class: "muted", text: "· ✓/✗ answer · number = position in phase" })),
      ),
      h("div", { class: "image-order" }, grid.element, details),
    ),
    card("PCA stimulus plot", plotHost),
    card(`Trials (${trials.length})`, trials.length ? trialTable(trials) : h("p", { class: "muted", text: "No answered trials yet." })),
  );
}
