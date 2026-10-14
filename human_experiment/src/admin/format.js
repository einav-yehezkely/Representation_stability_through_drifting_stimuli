// Small view helpers shared by the admin page modules.

import { h } from "../components/dom.js";
import { csvFileName, serializeSessionCsv } from "../services/dataStorage/csvSerializer.js";

export const STATUS_LABELS = {
  complete: "Complete",
  in_progress: "In progress",
  interrupted: "Interrupted",
  incomplete_dev_early_finish: "Ended early (dev)",
};

export const RESPONSE_SHORT = { A: "A", "Probably A": "pA", "Probably B": "pB", B: "B" };

export function percent(numerator, denominator) {
  return denominator ? `${Math.round((100 * numerator) / denominator)}%` : "—";
}

export function formatDateTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function formatRelative(iso, now = Date.now()) {
  if (!iso) return "—";
  const seconds = Math.round((now - Date.parse(iso)) / 1000);
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

export function formatDuration(startIso, endIso) {
  const ms = Date.parse(endIso) - Date.parse(startIso);
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const minutes = Math.round(ms / 60000);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export function formatMs(ms) {
  return Number.isFinite(ms) ? `${Math.round(ms)} ms` : "—";
}

export function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function statusBadge(status) {
  return h("span", { class: `badge badge-${status ?? "unknown"}` }, STATUS_LABELS[status] ?? status ?? "Unknown");
}

export function statTile(label, value, note = null) {
  return h("div", { class: "stat" }, h("span", { class: "stat-label", text: label }), h("span", { class: "stat-value", text: value }), note ? h("span", { class: "stat-note", text: note }) : null);
}

export function card(title, ...children) {
  return h("section", { class: "card" }, title ? h("h2", { class: "card-title", text: title }) : null, ...children);
}

/** CSV of one session, with exactly the columns of the participant CSV and `npm run data:csv`. */
export function sessionCsv(sessionId, meta, trials) {
  const incomplete = meta.completionStatus !== "complete";
  return { name: csvFileName(sessionId, { incomplete }), text: serializeSessionCsv(meta, trials, { incomplete }) };
}
