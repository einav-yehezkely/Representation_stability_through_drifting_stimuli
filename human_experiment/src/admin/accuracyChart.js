// Rolling accuracy over the answered trials of one session (single series, so no legend box:
// the card title names it). Chance level (50%) and the start of drift are marked. A crosshair
// snaps to the nearest trial and a tooltip shows its values; the trial table below the chart
// holds every value without hovering.

import { h } from "../components/dom.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const HEIGHT = 240;
const PAD = { top: 18, right: 16, bottom: 34, left: 44 };

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Accuracy over the last `window` answered trials (fewer at the start), per trial. */
export function rollingAccuracy(records, window) {
  let correct = 0;
  return records.map((record, i) => {
    if (record.correct === true) correct += 1;
    if (i >= window && records[i - window].correct === true) correct -= 1;
    const n = Math.min(i + 1, window);
    return { record, accuracy: correct / n, n };
  });
}

function niceStep(span, target) {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? raw;
}

export function createAccuracyChart({ records, window = 20 }) {
  const points = rollingAccuracy(records, window);
  const tooltip = h("div", { class: "chart-tooltip", hidden: true });
  const host = h("div", { class: "chart" });
  const element = h("div", { class: "chart-wrap" }, host, tooltip);
  if (!points.length) {
    host.append(h("p", { class: "muted", text: "No answered trials yet." }));
    return { element };
  }

  function render(width) {
    const w = Math.max(320, width);
    const innerW = w - PAD.left - PAD.right;
    const innerH = HEIGHT - PAD.top - PAD.bottom;
    const n = points.length;
    const x = (i) => PAD.left + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const y = (v) => PAD.top + (1 - v) * innerH;

    const svg = el("svg", { viewBox: `0 0 ${w} ${HEIGHT}`, width: w, height: HEIGHT, role: "img", "aria-label": `Rolling accuracy over the last ${window} trials` });
    const grid = el("g", { class: "chart-grid" });
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      grid.append(el("line", { x1: PAD.left, x2: w - PAD.right, y1: y(v), y2: y(v) }));
      svg.append(el("text", { class: "chart-axis", x: PAD.left - 8, y: y(v) + 4, "text-anchor": "end" }, `${v * 100}%`));
    }
    svg.append(grid);

    // X ticks at clean trial numbers.
    const step = niceStep(n, Math.max(2, Math.floor(innerW / 90)));
    for (let t = step; t <= n; t += step) {
      svg.append(el("text", { class: "chart-axis", x: x(t - 1), y: HEIGHT - PAD.bottom + 18, "text-anchor": "middle" }, String(t)));
    }
    svg.append(el("text", { class: "chart-axis", x: PAD.left, y: HEIGHT - 4 }, "Trial"));

    // Chance level.
    svg.append(el("line", { class: "chart-ref", x1: PAD.left, x2: w - PAD.right, y1: y(0.5), y2: y(0.5) }));
    svg.append(el("text", { class: "chart-ref-label", x: w - PAD.right - 4, y: y(0.5) - 6, "text-anchor": "end" }, "chance"));

    // Start of drift.
    const firstDrift = points.findIndex((p) => p.record.phase === "drift");
    if (firstDrift > 0) {
      const xd = (x(firstDrift - 1) + x(firstDrift)) / 2;
      svg.append(el("rect", { class: "chart-phase", x: PAD.left, y: PAD.top, width: xd - PAD.left, height: innerH }));
      svg.append(el("line", { class: "chart-phase-line", x1: xd, x2: xd, y1: PAD.top, y2: HEIGHT - PAD.bottom }));
      svg.append(el("text", { class: "chart-phase-label", x: xd - 6, y: PAD.top + 12, "text-anchor": "end" }, "Training"));
      svg.append(el("text", { class: "chart-phase-label", x: xd + 6, y: PAD.top + 12 }, "Drift (no feedback)"));
    }

    const path = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.accuracy).toFixed(1)}`).join("");
    svg.append(el("path", { class: "chart-line", d: path }));
    const last = points.at(-1);
    svg.append(el("circle", { class: "chart-dot", cx: x(n - 1), cy: y(last.accuracy), r: 4 }));

    // Hover layer: crosshair snaps to the nearest trial.
    const cross = el("line", { class: "chart-cross", y1: PAD.top, y2: HEIGHT - PAD.bottom, visibility: "hidden" });
    const marker = el("circle", { class: "chart-dot", r: 4.5, visibility: "hidden" });
    const overlay = el("rect", { x: PAD.left, y: 0, width: innerW, height: HEIGHT, fill: "transparent", tabindex: 0 });
    svg.append(cross, marker, overlay);

    const showAt = (i) => {
      const p = points[i];
      const r = p.record;
      cross.setAttribute("x1", x(i));
      cross.setAttribute("x2", x(i));
      marker.setAttribute("cx", x(i));
      marker.setAttribute("cy", y(p.accuracy));
      cross.setAttribute("visibility", "visible");
      marker.setAttribute("visibility", "visible");
      tooltip.replaceChildren(
        h("strong", { text: `${Math.round(p.accuracy * 100)}%` }),
        h("span", { text: ` correct in the last ${p.n} trials` }),
        h("br"),
        h("span", { class: "muted", text: `Trial ${r.trialNumber} · ${r.phase} ${r.phaseTrialNumber} · center A ${r.currentAngleA}° · ${r.correct ? "correct" : "incorrect"}` }),
      );
      tooltip.hidden = false;
      const left = Math.min(Math.max(x(i) + 12, 0), w - tooltip.offsetWidth - 4);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${Math.max(0, y(p.accuracy) - 52)}px`;
    };
    const hide = () => {
      cross.setAttribute("visibility", "hidden");
      marker.setAttribute("visibility", "hidden");
      tooltip.hidden = true;
    };
    let focused = n - 1;
    overlay.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * w;
      focused = Math.round(((px - PAD.left) / innerW) * (n - 1));
      showAt(Math.min(n - 1, Math.max(0, focused)));
    });
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("focus", () => showAt(focused));
    overlay.addEventListener("blur", hide);
    overlay.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      focused = Math.min(n - 1, Math.max(0, focused + (event.key === "ArrowRight" ? 1 : -1)));
      showAt(focused);
    });
    host.replaceChildren(svg);
  }

  // Redraw at the real width so text and lines stay crisp at every size.
  let lastWidth = 0;
  const observer = new ResizeObserver(([entry]) => {
    const width = Math.round(entry.contentRect.width);
    if (width && Math.abs(width - lastWidth) > 4) {
      lastWidth = width;
      render(width);
    }
  });
  observer.observe(host);
  return { element };
}
