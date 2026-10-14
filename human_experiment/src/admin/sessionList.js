// Admin: all sessions, with filters, totals and a combined CSV download.

import { h } from "../components/dom.js";
import { CSV_COLUMNS } from "../services/dataStorage/csvSerializer.js";
import { browserDownload } from "../services/dataStorage/browserDownload.js";
import { card, formatDateTime, formatMs, formatRelative, percent, sessionCsv, statTile, statusBadge, STATUS_LABELS } from "./format.js";

const REFRESH_MS = 30000;

export function createSessionList({ source, onOpen, onError }) {
  const state = { sessions: [], query: "", status: "all", includeDev: false, loadedAt: null };

  const stats = h("div", { class: "stats" });
  const search = h("input", {
    type: "search",
    placeholder: "Search participant or session ID",
    "aria-label": "Search",
    oninput: (e) => ((state.query = e.target.value.trim().toLowerCase()), render()),
  });
  const statusSelect = h(
    "select",
    { "aria-label": "Status", onchange: (e) => ((state.status = e.target.value), render()) },
    h("option", { value: "all" }, "All statuses"),
    Object.entries(STATUS_LABELS).map(([value, label]) => h("option", { value }, label)),
  );
  const devToggle = h("input", { type: "checkbox", onchange: (e) => ((state.includeDev = e.target.checked), render()) });
  const updated = h("span", { class: "muted small" });
  const refreshButton = h("button", { type: "button", class: "secondary small", onclick: () => refresh() }, "Refresh");
  const exportButton = h("button", { type: "button", class: "small", onclick: () => exportAll() }, "Download CSV (shown sessions)");
  const tableBody = h("tbody");
  const table = h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      { class: "data-table sessions-table" },
      h(
        "thead",
        {},
        h("tr", {}, ["Started", "Participant", "Status", "Training", "Drift", "Drift accuracy", "Mean RT", "Last activity"].map((t) => h("th", { text: t }))),
      ),
      tableBody,
    ),
  );
  const empty = h("p", { class: "muted empty", hidden: true });

  const element = h(
    "div",
    { class: "view" },
    stats,
    h("div", { class: "filters" }, search, statusSelect, h("label", { class: "check" }, devToggle, " Include development sessions"), h("span", { class: "spacer" }), updated, refreshButton, exportButton),
    card(null, table, empty),
  );

  function visible() {
    return state.sessions.filter(
      (s) =>
        (state.includeDev || !s.developmentMode) &&
        (state.status === "all" || s.status === state.status) &&
        (!state.query || `${s.participantId ?? ""} ${s.sessionId}`.toLowerCase().includes(state.query)),
    );
  }

  function render() {
    const base = state.sessions.filter((s) => state.includeDev || !s.developmentMode);
    const count = (status) => base.filter((s) => s.status === status).length;
    stats.replaceChildren(
      statTile("Sessions", String(base.length), state.includeDev ? "including development" : "participants only"),
      statTile("Complete", String(count("complete"))),
      statTile("In progress", String(count("in_progress")), "active in the last 30 min"),
      statTile("Interrupted", String(count("interrupted"))),
    );
    const rows = visible();
    exportButton.disabled = rows.length === 0;
    empty.hidden = rows.length > 0;
    empty.textContent = state.sessions.length ? "No sessions match the filters." : "No sessions have been recorded yet.";
    tableBody.replaceChildren(
      ...rows.map((s) => {
        const open = () => onOpen(s.sessionId);
        const planned = s.driftTrialsPlanned ?? "?";
        return h(
          "tr",
          { tabindex: 0, onclick: open, onkeydown: (e) => (e.key === "Enter" ? open() : null) },
          h("td", { text: formatDateTime(s.sessionStartTime) }),
          h(
            "td",
            {},
            h("span", { class: "mono", text: s.participantId ?? "—" }),
            s.developmentMode ? h("span", { class: "tag", text: "dev" }) : null,
            h("div", { class: "muted small mono", text: s.sessionId.slice(0, 8) }),
          ),
          h("td", {}, statusBadge(s.status)),
          h(
            "td",
            {},
            `${s.trainingTrials} trials`,
            h("div", { class: "muted small", text: s.criterionMetAtTrial ? `criterion met at trial ${s.criterionMetAtTrial}` : s.trainingWindowAccuracy !== null ? `last window ${Math.round(s.trainingWindowAccuracy * 100)}%` : "window not full yet" }),
          ),
          h("td", {}, `${s.driftTrials} / ${planned}`, h("div", { class: "progress", "aria-hidden": "true" }, h("span", { style: { width: `${planned === "?" ? 0 : Math.min(100, (100 * s.driftTrials) / planned)}%` } }))),
          h("td", { class: "num", text: percent(s.driftCorrect, s.driftTrials) }),
          h("td", { class: "num", text: formatMs(s.meanResponseTimeMs) }),
          h("td", { text: formatRelative(s.lastActivity) }),
        );
      }),
    );
    updated.textContent = state.loadedAt ? `Updated ${new Date(state.loadedAt).toLocaleTimeString()}` : "";
  }

  async function refresh() {
    element.classList.add("is-loading"); // keep the previous table visible while reloading
    try {
      state.sessions = await source.listSessions();
      state.loadedAt = Date.now();
      render();
    } catch (error) {
      onError(error);
    } finally {
      element.classList.remove("is-loading");
    }
  }

  async function exportAll() {
    const rows = visible();
    exportButton.disabled = true;
    const label = exportButton.textContent;
    const header = `${CSV_COLUMNS.join(",")}\r\n`;
    const parts = [header];
    try {
      for (const [i, s] of rows.entries()) {
        exportButton.textContent = `Preparing ${i + 1} / ${rows.length}…`;
        const { meta, trials } = await source.getSession(s.sessionId);
        parts.push(sessionCsv(s.sessionId, meta, trials).text.slice(header.length));
      }
      browserDownload(`sessions_${new Date().toISOString().slice(0, 10)}.csv`, parts.join(""));
    } catch (error) {
      onError(error);
    } finally {
      exportButton.textContent = label;
      exportButton.disabled = false;
    }
  }

  const timer = setInterval(() => {
    if (element.isConnected && document.visibilityState === "visible") refresh();
  }, REFRESH_MS);

  return {
    element,
    refresh,
    destroy: () => clearInterval(timer),
  };
}
