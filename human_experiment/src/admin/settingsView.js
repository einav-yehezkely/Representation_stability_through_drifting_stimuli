// Admin: experiment settings (initial center angles, drift movement). Saved settings apply to
// sessions that start afterwards; running sessions keep the settings they started with.

import { h } from "../components/dom.js";
import { applySettings, findSettingsProblems, normalizeAngle, oppositeAngle, trialsForFullRotation } from "../config/experimentSettings.js";
import { collectNearest } from "../scientific/clusterConstruction.js";
import { angleDeg, initialCenters } from "../scientific/trajectory.js";
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

/**
 * The example face of each group: exactly the first face a participant sees for that group in
 * training. It uses the experiment's own selection: the starting centers of initialCenters (the
 * source rule: closest angle with a radius near scientific.targetRadius, B = -A) and the nearest
 * image to each center (selectStimulus with clusterSize 1). Neighboring angles can share a face
 * when no closer image exists on the trajectory, just as in the experiment.
 */
export function exampleFaces(pca, baseConfig, values) {
  const centers = initialCenters(pca, applySettings(baseConfig, values));
  return Object.fromEntries(
    [
      ["A", centers.centerA],
      ["B", centers.centerB],
    ].map(([group, center]) => {
      const [{ index }] = collectNearest(center, pca, 1);
      const [x, y] = [pca.pc1[index], pca.pc2[index]];
      return [group, { imageId: pca.imageIds[index], angle: angleDeg(x, y), radius: Math.hypot(x, y) }];
    }),
  );
}

/**
 * Image URLs of the example faces (both groups) for every whole degree of center A, ordered by
 * distance from `aroundDegree` (nearest first), so the faces around the current setting load first.
 */
export function circleFaceUrls(pca, baseConfig, facesDirectory, aroundDegree = 0) {
  const around = normalizeAngle(Math.round(aroundDegree));
  const distance = (d) => Math.min(Math.abs(d - around), 360 - Math.abs(d - around));
  const degrees = Array.from({ length: 360 }, (_, d) => d).sort((x, y) => distance(x) - distance(y) || x - y);
  const urls = degrees.flatMap((d) => {
    const { A, B } = exampleFaces(pca, baseConfig, { "training.initialAngleA": d, "training.initialAngleB": oppositeAngle(d) });
    return [A, B].map((face) => `${facesDirectory}${encodeURIComponent(face.imageId)}`);
  });
  return [...new Set(urls)];
}

const PRELOAD_CONCURRENCY = 6;

export function createSettingsView({ source, defaults, getPca, facesDirectory, onSaved, onError }) {
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

  // Example face of each group, updated whenever the starting angles change.
  const faceCard = (group) => {
    const img = h("img", { alt: "", width: 134, height: 164, decoding: "async" });
    const caption = h("span", { class: "muted small" });
    return { img, caption, element: h("figure", { class: "example-face", "data-group": group }, h("span", { class: "example-face-frame" }, img, h("span", { class: "example-face-group", text: group })), h("figcaption", {}, h("strong", { text: `Group ${group} example` }), caption)) };
  };
  const faces = { A: faceCard("A"), B: faceCard("B") };
  for (const { img, element } of Object.values(faces)) {
    img.addEventListener("load", () => element.classList.remove("is-loading"));
    img.addEventListener("error", () => element.classList.remove("is-loading"));
  }

  // Faces for the whole circle, downloaded in the background once the PCA data is available.
  // The loaded Image objects are kept, so showing one of them later is immediate.
  const preloadStatus = h("span", { class: "muted small preload-status" });
  const preloaded = [];
  let preloadStarted = false;
  function preloadCircle(values) {
    const pca = getPca();
    if (preloadStarted || !pca || !baseConfig) return;
    preloadStarted = true;
    const a = values["training.initialAngleA"];
    const queue = circleFaceUrls(pca, baseConfig, facesDirectory, Number.isFinite(a) ? a : 0);
    const total = queue.length;
    let done = 0;
    preloadStatus.textContent = `Preparing faces for the whole circle: 0 / ${total}`;
    const next = () => {
      const url = queue.shift();
      if (!url) return;
      const image = new Image();
      preloaded.push(image);
      const finish = () => {
        done += 1;
        preloadStatus.textContent = done < total ? `Preparing faces for the whole circle: ${done} / ${total}` : "Faces for every degree are ready.";
        next();
      };
      image.onload = () => image.decode().catch(() => null).then(finish);
      image.onerror = finish;
      image.src = url;
    };
    // Detached images delay the page's load event, so start only once the page has loaded.
    const start = () => {
      for (let i = 0; i < PRELOAD_CONCURRENCY; i += 1) next();
    };
    if (document.readyState === "complete") start();
    else window.addEventListener("load", start, { once: true });
  }

  function updateFaces(values) {
    const pca = getPca();
    let examples = null;
    if (pca && baseConfig && Number.isFinite(values["training.initialAngleA"]) && Number.isFinite(values["training.initialAngleB"])) {
      try {
        examples = exampleFaces(pca, baseConfig, values);
      } catch {
        examples = null; // invalid settings: reported by the validation on save
      }
    }
    for (const group of ["A", "B"]) {
      const { img, caption, element } = faces[group];
      const example = examples?.[group];
      element.classList.toggle("is-empty", !example);
      if (!example) {
        img.removeAttribute("src");
        caption.textContent = pca ? "Enter a valid angle." : "Loading faces…";
        continue;
      }
      const src = `${facesDirectory}${encodeURIComponent(example.imageId)}`;
      if (img.getAttribute("src") !== src) {
        img.setAttribute("src", src);
        // Until the new picture has loaded, the old one is dimmed so it never sits under the new name.
        element.classList.toggle("is-loading", !img.complete);
      }
      img.alt = `Example face of group ${group}: ${example.imageId}`;
      caption.textContent = `${example.imageId} · at ${example.angle.toFixed(1)}°, radius ${example.radius.toFixed(3)}`;
    }
  }

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
    updateFaces(v);
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
          h("div", { class: "settings-row example-faces" }, faces.A.element, faces.B.element),
          preloadStatus,
          h("p", { class: "muted small", text: "Each example is the first face a participant sees for that group in training, chosen exactly as in the experiment: the starting center is the image closest to the angle with a radius near 0.45 (the rule of the computational experiment), B is opposite A, and each group shows the image nearest to its center. When no closer image exists, neighboring angles show the same face, as they would for participants." }),
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
      preloadCircle(read());
    },
  };
}

