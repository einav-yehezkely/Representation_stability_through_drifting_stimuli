// Researcher review page (not linked from the participant experiment).
// Shows the PCA stimulus-selection plot for a session saved in this browser's localStorage
// (live-updating while the experiment runs in another tab) or for an exported CSV file.

import "./styles.css";
import "./visualization/plot.css";
import { h } from "./components/dom.js";
import { experimentConfig } from "./config/experimentConfig.js";
import { loadPca } from "./scientific/pcaLoader.js";
import { parseCsv } from "./services/dataStorage/csvSerializer.js";
import { LocalCsvStorage } from "./services/dataStorage/localCsvStorage.js";
import { createPlotViewer } from "./visualization/plotViewer.js";

const NUMERIC = ["trialNumber", "phaseTrialNumber", "pc1", "pc2", "stimulusAngle", "currentAngleA", "currentAngleB", "responseTimeMs", "imageIndex", "distanceToCenter", "centerA_pc1", "centerA_pc2", "centerB_pc1", "centerB_pc2"];

function recordsFromCsv(rows) {
  return rows.map((row) => {
    const r = { ...row };
    for (const key of NUMERIC) r[key] = row[key] === "" ? null : Number(row[key]);
    r.correct = row.correct === "true";
    return r;
  });
}

function latestCenters(records) {
  const last = records.at(-1);
  if (!last) return null;
  return { centerA: [last.centerA_pc1, last.centerA_pc2], centerB: [last.centerB_pc1, last.centerB_pc2] };
}

async function boot() {
  const root = document.getElementById("review");
  root.style.cssText = "max-width:1100px;margin:0 auto;padding:24px 16px;";
  root.append(h("p", { text: "Loading PCA data…" }));
  const pca = await loadPca(experimentConfig);
  const storage = new LocalCsvStorage({ prefix: experimentConfig.storage.localStoragePrefix, download: () => {} });

  let viewer = null;
  let viewerConfigKey = null;
  let selected = null; // { kind: "local", sessionId } | { kind: "csv", ... }

  const sessionSelect = h("select", { onchange: (e) => selectLocal(e.target.value) });
  const fileInput = h("input", { type: "file", accept: ".csv,text/csv", onchange: (e) => loadCsvFile(e.target.files[0]) });
  const summary = h("p", { class: "status-line" });
  const host = h("div");

  root.replaceChildren(
    h("h1", { text: "Researcher review — PCA stimulus selection" }),
    h(
      "p",
      { class: "status-line" },
      "For researchers only. Do not show this page to participants: it reveals group centers and group membership.",
    ),
    h("div", { class: "plot-controls" }, h("label", {}, "Saved session ", sessionSelect), h("label", {}, "or exported CSV ", fileInput)),
    summary,
    host,
  );

  async function refreshSessionList() {
    const sessions = await storage.listSessions();
    const current = sessionSelect.value;
    sessionSelect.replaceChildren(
      h("option", { value: "" }, sessions.length ? "— choose —" : "— none saved in this browser —"),
      ...sessions
        .slice()
        .reverse()
        .map((s) =>
          h(
            "option",
            { value: s.sessionId },
            `${s.meta?.sessionStartTime ?? "?"} · ${s.sessionId.slice(0, 8)} · ${s.meta?.completionStatus ?? "unreadable"}${s.meta?.developmentMode ? " · dev" : ""}`,
          ),
        ),
    );
    sessionSelect.value = current;
  }

  function show({ sessionId, meta, records }) {
    const config = meta?.effectiveConfig ?? experimentConfig;
    const key = JSON.stringify([config.training, config.drift, config.scientific]);
    if (!viewer || key !== viewerConfigKey) {
      viewer = createPlotViewer({ pca, config, facesDirectory: experimentConfig.assets.facesDirectory });
      viewerConfigKey = key;
      host.replaceChildren(viewer.element);
    }
    const last = records.at(-1) ?? null;
    const centers = latestCenters(records);
    summary.textContent = `${records.length} answered trials (${records.filter((r) => r.phase === "training").length} training, ${records.filter((r) => r.phase === "drift").length} drift) · status: ${meta?.completionStatus ?? "unknown"}. Centers and the highlighted stimulus are those of the most recent answered trial.`;
    viewer.setView({
      sessionId,
      records,
      centers,
      current: last,
      title: last ? `Session ${sessionId.slice(0, 8)} · trial ${last.trialNumber} (${last.phase}) · A ${last.currentAngleA}° / B ${last.currentAngleB}°` : `Session ${sessionId.slice(0, 8)}`,
    });
  }

  async function selectLocal(sessionId) {
    if (!sessionId) return;
    selected = { kind: "local", sessionId };
    storage.refresh(sessionId);
    const { meta, trials } = await storage.getSession(sessionId);
    show({ sessionId, meta, records: trials });
  }

  async function loadCsvFile(file) {
    if (!file) return;
    const rows = parseCsv(await file.text());
    if (!rows.length) {
      summary.textContent = "The CSV file contains no trials.";
      return;
    }
    const meta = {
      completionStatus: rows[0].completionStatus,
      effectiveConfig: rows[0].effectiveConfigJson ? JSON.parse(rows[0].effectiveConfigJson) : null,
    };
    selected = { kind: "csv" };
    sessionSelect.value = "";
    show({ sessionId: rows[0].sessionId, meta, records: recordsFromCsv(rows) });
  }

  // Live updates while the experiment runs in another tab of the same browser.
  window.addEventListener("storage", async (event) => {
    if (!event.key?.startsWith(`${experimentConfig.storage.localStoragePrefix}:`)) return;
    await refreshSessionList();
    if (selected?.kind === "local" && event.key.includes(selected.sessionId)) await selectLocal(selected.sessionId);
  });

  await refreshSessionList();
}

boot();
