// Admin: experiment settings (initial center angles, drift movement). Saved settings apply to
// sessions that start afterwards; running sessions keep the settings they started with.

import { h } from "../components/dom.js";
import { findSettingsProblems, oppositeAngle, trialsForFullRotation } from "../config/experimentSettings.js";
import { card, formatDateTime } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const round = (v, digits = 6) => Number(Number(v).toFixed(digits));

function svgEl(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Start positions of the two centers on the circle and the direction of movement (CCW, as in the PCA plot). */
function diagram(angleA, angleB, degrees) {
  const size = 240;
  const c = size / 2;
  const r = 78;
  const point = (deg, radius = r) => [c + radius * Math.cos((deg * Math.PI) / 180), c - radius * Math.sin((deg * Math.PI) / 180)];
  const svg = svgEl("svg", { viewBox: `0 0 ${size} ${size}`, width: size, height: size, role: "img", "aria-label": `Center A starts at ${angleA}°, center B at ${angleB}°` });
  svg.append(svgEl("line", { x1: c - r - 18, y1: c, x2: c + r + 18, y2: c, class: "diagram-axis" }));
  svg.append(svgEl("line", { x1: c, y1: c - r - 18, x2: c, y2: c + r + 18, class: "diagram-axis" }));
  svg.append(svgEl("circle", { cx: c, cy: c, r, class: "diagram-circle" }));
  for (const [deg, label] of [[0, "0°"], [90, "90°"], [180, "180°"], [270, "270°"]]) {
    const [x, y] = point(deg, r + 30);
    svg.append(svgEl("text", { x, y: y + 4, "text-anchor": "middle", class: "diagram-tick" }, label));
  }
  // Direction of movement: an arc from A, counterclockwise.
  if (Number.isFinite(angleA) && degrees > 0) {
    const sweep = 40;
    const [x1, y1] = point(angleA + 8, r + 12);
    const [x2, y2] = point(angleA + 8 + sweep, r + 12);
    svg.append(svgEl("path", { d: `M${x1},${y1} A${r + 12},${r + 12} 0 0 0 ${x2},${y2}`, class: "diagram-arrow", "marker-end": "url(#arrowhead)" }));
    const defs = svgEl("defs");
    const marker = svgEl("marker", { id: "arrowhead", viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
    marker.append(svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: "diagram-arrowhead" }));
    defs.append(marker);
    svg.prepend(defs);
  }
  for (const [deg, group] of [[angleA, "A"], [angleB, "B"]]) {
    if (!Number.isFinite(deg)) continue;
    const [x, y] = point(deg);
    svg.append(svgEl("line", { x1: c, y1: c, x2: x, y2: y, class: `diagram-spoke group-${group}` }));
    svg.append(svgEl("circle", { cx: x, cy: y, r: 11, class: `diagram-dot group-${group}` }));
    svg.append(svgEl("text", { x, y: y + 4, "text-anchor": "middle", class: "diagram-dot-label" }, group));
  }
  return svg;
}

export function createSettingsView({ source, defaults, getPca, onSaved, onError }) {
  let saved = null; // { values, updatedAt } | null
  let baseConfig = null; // experimentConfig, for validation (set by load)
  const field = (label, { unit, ...attrs }, hint) => {
    const input = h("input", { type: "number", step: "any", ...attrs });
    return {
      input,
      element: h(
        "label",
        { class: "setting" },
        h("span", { class: "setting-label", text: label }),
        h("span", { class: "setting-input" }, input, h("span", { class: "unit", text: unit })),
        hint ? h("span", { class: "muted small", text: hint }) : null,
      ),
    };
  };
  const angleA = field("Center A — initial angle", { "data-setting": "training.initialAngleA", unit: "°" });
  const angleB = field("Center B — initial angle", { "data-setting": "training.initialAngleB", unit: "°" }, "Always 180° from center A: changing one angle moves the other.");
  const degrees = field("Movement per drift trial", { "data-setting": "drift.degreesPerTrial", min: 0.01, max: 360, unit: "°" }, "Both centers rotate counterclockwise by this amount after every drift answer.");
  const trials = field("Number of drift trials", { "data-setting": "drift.numberOfTrials", min: 1, step: 1, unit: "trials" });

  const rotationNote = h("p", { class: "rotation-note" });
  const diagramHost = h("div", { class: "settings-diagram" });
  const problemsBox = h("ul", { class: "form-problems" });
  const status = h("p", { class: "muted small settings-status" });
  const saveButton = h("button", { type: "button", onclick: () => save() }, "Save settings");
  const resetButton = h("button", { type: "button", class: "secondary", onclick: () => fill(defaults) }, "Reset to defaults");

  const read = () => ({
    "training.initialAngleA": Number.parseFloat(angleA.input.value),
    "training.initialAngleB": Number.parseFloat(angleB.input.value),
    "drift.degreesPerTrial": Number.parseFloat(degrees.input.value),
    "drift.numberOfTrials": Number.parseInt(trials.input.value, 10),
  });

  function fill(values) {
    angleA.input.value = String(values["training.initialAngleA"]);
    angleB.input.value = String(values["training.initialAngleB"]);
    degrees.input.value = String(values["drift.degreesPerTrial"]);
    trials.input.value = String(values["drift.numberOfTrials"]);
    update();
  }

  function update() {
    const v = read();
    const total = round(v["drift.numberOfTrials"] * v["drift.degreesPerTrial"], 4);
    if (Number.isFinite(total)) {
      const full = Math.abs(total - 360) < 1e-6;
      rotationNote.className = `rotation-note${full ? "" : " is-warning"}`;
      rotationNote.textContent = full
        ? `${v["drift.numberOfTrials"]} trials × ${v["drift.degreesPerTrial"]}° = one full rotation (360°).`
        : `${v["drift.numberOfTrials"]} trials × ${v["drift.degreesPerTrial"]}° = ${total}°, not a full rotation${360 % v["drift.degreesPerTrial"] ? ` (360 is not a multiple of ${v["drift.degreesPerTrial"]}°)` : ""}.`;
    } else rotationNote.textContent = "";
    diagramHost.replaceChildren(diagram(v["training.initialAngleA"], v["training.initialAngleB"], v["drift.degreesPerTrial"]));
    const current = saved?.values ?? defaults;
    const dirty = Object.keys(current).some((k) => current[k] !== v[k]);
    status.textContent = `${saved ? `Saved ${formatDateTime(saved.updatedAt)}.` : "Default settings (nothing saved yet)."}${dirty ? " · Unsaved changes" : ""}`;
    status.classList.toggle("is-dirty", dirty);
    problemsBox.replaceChildren();
  }

  // Linked fields: B stays opposite A; the trial count follows the step to complete a full rotation.
  angleA.input.addEventListener("input", () => {
    const a = Number.parseFloat(angleA.input.value);
    if (Number.isFinite(a)) angleB.input.value = String(round(oppositeAngle(a)));
    update();
  });
  angleB.input.addEventListener("input", () => {
    const b = Number.parseFloat(angleB.input.value);
    if (Number.isFinite(b)) angleA.input.value = String(round(oppositeAngle(b)));
    update();
  });
  degrees.input.addEventListener("input", () => {
    const n = trialsForFullRotation(Number.parseFloat(degrees.input.value));
    if (n !== null) trials.input.value = String(n);
    update();
  });
  trials.input.addEventListener("input", update);

  async function save() {
    const values = read();
    const problems = findSettingsProblems(values, baseConfig, { pcaCount: getPca()?.count });
    if (problems.length) {
      problemsBox.replaceChildren(...problems.map((p) => h("li", { text: p })));
      return;
    }
    saveButton.disabled = true;
    try {
      saved = await source.saveSettings(values);
      update();
      status.textContent = `Saved ${formatDateTime(saved.updatedAt)}. New sessions use these settings.`;
      onSaved(saved);
    } catch (error) {
      onError(error);
    } finally {
      saveButton.disabled = false;
    }
  }

  const element = h(
    "div",
    { class: "view" },
    h("h1", { text: "Experiment settings" }),
    h(
      "p",
      { class: "muted intro" },
      "These settings apply to every session that starts after you save them. Sessions already running keep the settings they started with, and every session records the settings it used. Changing settings during data collection mixes conditions, so note when you change them.",
    ),
    card(
      null,
      h(
        "div",
        { class: "settings-layout" },
        h(
          "div",
          { class: "settings-form" },
          h("h2", { class: "card-title", text: "Starting positions" }),
          h("div", { class: "settings-row" }, angleA.element, angleB.element),
          h("h2", { class: "card-title", text: "Drift" }),
          h("div", { class: "settings-row" }, degrees.element, trials.element),
          rotationNote,
          h("p", { class: "muted small", text: "The number of trials is filled in automatically for one full rotation (360 ÷ movement per trial) whenever you change the movement. You can still edit it." }),
          problemsBox,
          h("div", { class: "actions" }, saveButton, resetButton),
          status,
        ),
        diagramHost,
      ),
    ),
  );

  return {
    element,
    /** Load the saved settings (or show the defaults). `baseConfig` is experimentConfig. */
    async load(config) {
      baseConfig = config;
      try {
        saved = await source.getSettings();
      } catch (error) {
        onError(error);
      }
      fill(saved?.values ?? defaults);
    },
  };
}

