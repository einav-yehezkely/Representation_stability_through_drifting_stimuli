// Admin: one session. Summary, accuracy over trials, the images in the order they were shown,
// the PCA stimulus plot and every trial record.

import { h } from "../components/dom.js";
import { browserDownload } from "../services/dataStorage/browserDownload.js";
import { createPlotViewer } from "../visualization/plotViewer.js";
import { createAccuracyChart } from "./accuracyChart.js";
import { card, formatDateTime, formatDuration, formatMs, median, percent, sessionCsv, statTile, statusBadge } from "./format.js";
import { createImageGrid, imageDetails } from "./imageGrid.js";

const ROLLING_WINDOW = 20;

const fixed = (v, digits) => (v === null || v === undefined || v === "" ? "—" : Number(v).toFixed(digits));

function trialItem(r) {
  return {
    trialNumber: r.trialNumber,
    sampleGroup: r.sampleGroup,
    sampleImageId: r.sampleImageId,
    optionImageIds: { A: r.optionAImageId, B: r.optionBImageId },
    leftGroup: r.leftGroup,
    chosenGroup: r.chosenGroup,
    correct: r.correct,
    record: r,
  };
}

function trialTable(trials) {
  const columns = [
    ["#", (r) => r.trialNumber, "num"],
    ["Sample", (r) => r.sampleImageId, "mono"],
    ["Sample group", (r) => r.sampleGroup],
    ["From trial", (r) => r.sampleSourceTrial ?? "—", "num"],
    ["Option A", (r) => r.optionAImageId, "mono"],
    ["Option B", (r) => r.optionBImageId, "mono"],
    ["Left", (r) => r.leftGroup],
    ["Answer", (r) => `${r.response} (${r.chosenGroup})`],
    ["Correct", (r) => (r.correct ? "✓" : "✗")],
    ["RT (ms)", (r) => Math.round(r.responseTimeMs), "num"],
    ["Sample angle", (r) => `${fixed(r.sampleAngle, 1)}°`, "num"],
    ["Center A / B", (r) => `${r.currentAngleA}° / ${r.currentAngleB}°`, "num"],
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
  // Sessions recorded with another PCA file: their coordinates belong to that file, not to the
  // data loaded now (the plot's background and trajectory come from the current file).
  const otherPcaFile = Boolean(pca && meta.pcaSha256 && meta.pcaSha256 !== pca.sha256);
  const sameAxes = pca && !otherPcaFile ? pca.axes : null;
  const plannedTrials = effectiveConfig.drift?.numberOfTrials ?? null;
  const fromSample = (group) => trials.filter((t) => t.sampleGroup === group);

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
          ["Trial", String(r.trialNumber)],
          ["Sample", `${r.sampleImageId} · group ${r.sampleGroup}${r.sampleSourceTrial ? ` · from trial ${r.sampleSourceTrial}` : ""}`],
          ["Option A", r.optionAImageId],
          ["Option B", r.optionBImageId],
          ["Left / right", `${r.leftGroup} / ${r.leftGroup === "A" ? "B" : "A"}`],
          ["Answer", `${r.response} (group ${r.chosenGroup}) · ${r.correct ? "correct" : "incorrect"}`],
          ["Response time", formatMs(r.responseTimeMs)],
          ["Sample / blank shown", `${formatMs(r.sampleDurationMs)} / ${formatMs(r.blankDurationMs)}`],
          ["Empty screen before sample", formatMs(r.interTrialDurationMs)],
          [sameAxes ? `Sample ${sameAxes.join(", ")}` : "Sample PCA coordinates", `${fixed(r.samplePc1, 4)}, ${fixed(r.samplePc2, 4)}`],
          ["Sample angle", `${fixed(r.sampleAngle, 2)}°`],
          ["Centers (nominal)", `A ${r.currentAngleA}° · B ${r.currentAngleB}°`],
        ]),
      );
    },
  });
  const filters = { group: "all", outcome: "all" };
  const applyFilters = () =>
    grid.setItems(
      trials
        .filter(
          (r) =>
            (filters.group === "all" || r.sampleGroup === filters.group) &&
            (filters.outcome === "all" || (filters.outcome === "incorrect") !== r.correct),
        )
        .map(trialItem),
      { empty: trials.length ? "No trials match the filters." : "No answered trials yet." },
    );
  const filterSelect = (name, options) =>
    h(
      "select",
      { "aria-label": name, onchange: (e) => ((filters[name] = e.target.value), applyFilters()) },
      options.map(([value, label]) => h("option", { value }, label)),
    );
  applyFilters();

  // PCA plot of the samples, colored by the group the participant chose (same viewer as the
  // development panel, which colors by the sample's group).
  const plotHost = h("div", {});
  if (pca && trials.length) {
    const viewer = createPlotViewer({ pca, config: effectiveConfig, facesDirectory, colorBy: "prediction" });
    const last = trials.at(-1);
    viewer.setView({
      sessionId,
      records: trials,
      centers: { centerA: [last.centerA_pc1, last.centerA_pc2], centerB: [last.centerB_pc1, last.centerB_pc2] },
      current: last,
      title: `Session ${sessionId.slice(0, 8)} · trial ${last.trialNumber} · A ${last.currentAngleA}° / B ${last.currentAngleB}°`,
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
          `Settings: center A ${effectiveConfig.drift?.initialAngleA}° / B ${effectiveConfig.drift?.initialAngleB}° · ${effectiveConfig.drift?.degreesPerTrial}° per trial × ${plannedTrials} trials`,
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
    otherPcaFile
      ? h("p", { class: "warning", text: `This session was recorded with a different PCA file (${meta.pcaSource ?? "unknown"}) than the one loaded now (${pca.source}). Its images, angles and coordinates are those of its own file; the gray points and trajectory in the PCA plot are from the current file.` })
      : null,
    h(
      "div",
      { class: "stats" },
      statTile("Trials", `${trials.length} / ${plannedTrials ?? "?"}`),
      statTile("Accuracy", percent(trials.filter((t) => t.correct).length, trials.length), "chose the sample's group"),
      statTile("Accuracy, sample from A", percent(fromSample("A").filter((t) => t.correct).length, fromSample("A").length)),
      statTile("Accuracy, sample from B", percent(fromSample("B").filter((t) => t.correct).length, fromSample("B").length)),
      statTile("Median RT", formatMs(median(trials.map((t) => Number(t.responseTimeMs)))), "two faces shown → answer"),
    ),
    card(`Rolling accuracy (last ${ROLLING_WINDOW} trials)`, createAccuracyChart({ records: trials, window: ROLLING_WINDOW }).element),
    card(
      "Image order",
      h(
        "div",
        { class: "filters" },
        filterSelect("group", [["all", "Sample from both groups"], ["A", "Sample from A"], ["B", "Sample from B"]]),
        filterSelect("outcome", [["all", "All answers"], ["correct", "Correct only"], ["incorrect", "Incorrect only"]]),
        h("span", { class: "legend" }, h("span", { class: "key key-a", text: "A" }), "Group A", h("span", { class: "key key-b", text: "B" }), "Group B", h("span", { class: "muted", text: "· each trial: sample, then the left and right faces · ring = chosen face · ✓/✗ answer" })),
      ),
      h("div", { class: "image-order" }, grid.element, details),
    ),
    card("PCA plot of the samples", plotHost),
    card(`Trials (${trials.length})`, trials.length ? trialTable(trials) : h("p", { class: "muted", text: "No answered trials yet." })),
  );
}
