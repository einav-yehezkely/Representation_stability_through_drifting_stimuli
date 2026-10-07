// Researcher-facing PCA plot (PRD "Scientific stimulus-selection visualization").
//
// Reproduces the visual conventions of the source plotting code (copied parameters, not
// imported; originals in human_experiment/reference_source/):
//   create_prediction_scatter (classify_rotation_resnet50.py lines 807-907):
//     all points gray (s=5, alpha=0.3); A blue, B red; dashed black reference circle of radius
//     max(sqrt(x^2 + y^2)) * 1.05 (alpha 0.5); gray radial lines every 20° (lw 0.5, alpha 0.5)
//     with "<angle>°" labels at 1.05 * radius; black x=0 / y=0 axes; grid; axis equal;
//     axis labels PC1 / PC2.
//   plot_clusters_with_given_indices (generate_rotation_sequence.py lines 205-259):
//     base point as a black "x", opposite point as a black "*".
// Additions required by the PRD: the circular trajectory of the centers, trajectory membership
// points, presented stimuli highlighted, and the currently displayed stimulus distinguished.

const SVG_NS = "http://www.w3.org/2000/svg";

export const PLOT_STYLE = Object.freeze({
  size: 760,
  margin: 64,
  allPoints: { color: "gray", radius: 1.6, opacity: 0.3 }, // s=5, alpha=0.3
  trajectory: { color: "#4d4d4d", radius: 2.2, opacity: 0.75 },
  sourceSequence: { color: "#9467bd", radius: 2.2, opacity: 0.9 },
  groupColor: { A: "blue", B: "red" },
  presentedRadius: { training: 3.6, drift: 3.6 },
  referenceCircle: { color: "black", dash: "6 4", opacity: 0.5, scale: 1.05 },
  spokes: { color: "gray", width: 0.5, opacity: 0.5, everyDeg: 20 },
  centerCircle: { color: "#222", dash: "2 3", width: 1.2 },
  current: { stroke: "black", radius: 9, width: 2.5 },
});

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function starPath(cx, cy, r) {
  const points = [];
  for (let i = 0; i < 10; i += 1) {
    const radius = i % 2 === 0 ? r : r * 0.45;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    points.push(`${(cx + radius * Math.cos(a)).toFixed(2)},${(cy + radius * Math.sin(a)).toFixed(2)}`);
  }
  return `M${points.join("L")}Z`;
}

/**
 * Create a plot bound to one PCA dataset. The static layers (all points, reference circle,
 * spokes, axes, trajectory) are drawn once; `update()` redraws only the dynamic layer.
 */
export function createPcaPlot({ pca, trajectoryIndices = [], sourceSequenceIndices = [], title = "" }) {
  const S = PLOT_STYLE;
  const refRadius = pca.maxRadius * S.referenceCircle.scale;
  const extent = refRadius * 1.12;
  const inner = S.size - 2 * S.margin;
  const sx = (x) => S.margin + ((x + extent) / (2 * extent)) * inner;
  const sy = (y) => S.margin + ((extent - y) / (2 * extent)) * inner;
  const unit = inner / (2 * extent);

  const svg = el("svg", {
    xmlns: SVG_NS,
    viewBox: `0 0 ${S.size} ${S.size + 58}`,
    width: "100%",
    "font-family": "DejaVu Sans, Arial, sans-serif",
    "font-size": 11,
    role: "img",
  });
  svg.append(el("rect", { x: 0, y: 0, width: S.size, height: S.size + 58, fill: "white" }));
  const titleNode = el("text", { x: S.size / 2, y: 26, "text-anchor": "middle", "font-size": 14 }, title);
  svg.append(titleNode);

  // Grid (matplotlib default grid at major ticks).
  const grid = el("g", { stroke: "#b0b0b0", "stroke-width": 0.6, opacity: 0.6 });
  const tick = 0.2;
  for (let v = -Math.floor(extent / tick) * tick; v <= extent + 1e-9; v += tick) {
    grid.append(el("line", { x1: sx(v), y1: sy(-extent), x2: sx(v), y2: sy(extent) }));
    grid.append(el("line", { x1: sx(-extent), y1: sy(v), x2: sx(extent), y2: sy(v) }));
    const label = v.toFixed(1).replace("-0.0", "0.0");
    svg.append(el("text", { x: sx(v), y: sy(-extent) + 16, "text-anchor": "middle", fill: "#333" }, label));
    svg.append(el("text", { x: sx(-extent) - 6, y: sy(v) + 4, "text-anchor": "end", fill: "#333" }, label));
  }
  svg.append(grid);
  svg.append(el("rect", { x: sx(-extent), y: sy(extent), width: inner, height: inner, fill: "none", stroke: "black", "stroke-width": 0.8 }));

  // All points.
  const all = el("g", { fill: S.allPoints.color, opacity: S.allPoints.opacity, "data-layer": "all-points" });
  for (let i = 0; i < pca.count; i += 1) {
    all.append(el("circle", { cx: sx(pca.pc1[i]).toFixed(2), cy: sy(pca.pc2[i]).toFixed(2), r: S.allPoints.radius }));
  }
  svg.append(all);

  // Reference circle and spokes (create_prediction_scatter).
  svg.append(
    el("circle", {
      cx: sx(0),
      cy: sy(0),
      r: refRadius * unit,
      fill: "none",
      stroke: S.referenceCircle.color,
      "stroke-dasharray": S.referenceCircle.dash,
      opacity: S.referenceCircle.opacity,
    }),
  );
  const spokes = el("g", { stroke: S.spokes.color, "stroke-width": S.spokes.width, opacity: S.spokes.opacity });
  for (let deg = 0; deg < 360; deg += S.spokes.everyDeg) {
    const rad = (deg * Math.PI) / 180;
    const x = refRadius * Math.cos(rad);
    const y = refRadius * Math.sin(rad);
    spokes.append(el("line", { x1: sx(0), y1: sy(0), x2: sx(x), y2: sy(y) }));
    svg.append(el("text", { x: sx(x * 1.05), y: sy(y * 1.05) + 4, "text-anchor": "middle" }, `${deg}°`));
  }
  svg.append(spokes);
  svg.append(el("line", { x1: sx(-extent), y1: sy(0), x2: sx(extent), y2: sy(0), stroke: "black", "stroke-width": 1 }));
  svg.append(el("line", { x1: sx(0), y1: sy(-extent), x2: sx(0), y2: sy(extent), stroke: "black", "stroke-width": 1 }));
  svg.append(el("text", { x: S.size / 2, y: S.size - 14, "text-anchor": "middle", "font-size": 13 }, "PC1"));
  svg.append(el("text", { x: 18, y: S.size / 2, "text-anchor": "middle", "font-size": 13, transform: `rotate(-90 18 ${S.size / 2})` }, "PC2"));

  // Trajectory membership layers.
  const trajectory = el("g", { fill: S.trajectory.color, opacity: S.trajectory.opacity, "data-layer": "trajectory" });
  for (const i of trajectoryIndices) trajectory.append(el("circle", { cx: sx(pca.pc1[i]), cy: sy(pca.pc2[i]), r: S.trajectory.radius }));
  svg.append(trajectory);
  const sourceSeq = el("g", { fill: S.sourceSequence.color, opacity: S.sourceSequence.opacity, "data-layer": "source-sequence", display: "none" });
  for (const i of sourceSequenceIndices) sourceSeq.append(el("circle", { cx: sx(pca.pc1[i]), cy: sy(pca.pc2[i]), r: S.sourceSequence.radius }));
  svg.append(sourceSeq);

  const centerCircle = el("circle", { fill: "none", stroke: S.centerCircle.color, "stroke-dasharray": S.centerCircle.dash, "stroke-width": S.centerCircle.width });
  svg.append(centerCircle);

  const dynamic = el("g", { "data-layer": "dynamic" });
  svg.append(dynamic);

  // Legend.
  const legend = el("g", { transform: `translate(${S.margin + 8}, ${S.size + 8})` });
  const legendItems = [
    ["circle", S.allPoints.color, "All images", 0.6],
    ["circle", S.trajectory.color, "Trajectory candidates", 1],
    ["circle", S.groupColor.A, "Presented A", 1],
    ["circle", S.groupColor.B, "Presented B", 1],
    ["ring", "black", "Current stimulus", 1],
    ["x", "black", "Center A", 1],
    ["star", "black", "Center B", 1],
  ];
  let lx = 0;
  let ly = 0;
  legendItems.forEach(([shape, color, label, opacity], i) => {
    if (i === 4) {
      lx = 0;
      ly = 18;
    }
    const g = el("g", { transform: `translate(0, ${ly})` });
    legend.append(g);
    appendLegendItem(g, shape, color, label, opacity);
  });
  function appendLegendItem(legend, shape, color, label, opacity) {
    if (shape === "circle") legend.append(el("circle", { cx: lx + 5, cy: 6, r: 4, fill: color, opacity }));
    if (shape === "ring") legend.append(el("circle", { cx: lx + 5, cy: 6, r: 5, fill: "none", stroke: color, "stroke-width": 2 }));
    if (shape === "x") legend.append(el("path", { d: `M${lx + 1},2L${lx + 9},10M${lx + 9},2L${lx + 1},10`, stroke: color, "stroke-width": 2 }));
    if (shape === "star") legend.append(el("path", { d: starPath(lx + 5, 6, 6), fill: color }));
    legend.append(el("text", { x: lx + 14, y: 10 }, label));
    lx += 30 + label.length * 6.4;
  }
  svg.append(legend);

  /**
   * @param {object} view
   * @param {object[]} view.records   actual trial records (CSV/engine schema)
   * @param {{phase:string, group:string}} view.filters  "all" | "training" | "drift", "all" | "A" | "B"
   * @param {{centerA:number[], centerB:number[]}|null} view.centers  current centers
   * @param {object|null} view.current  { imageIndex, trueGroup } of the displayed stimulus
   * @param {boolean} view.showSourceSequence
   * @param {(record:object) => void} [view.onSelect]
   */
  function update({ records = [], filters = { phase: "all", group: "all" }, centers = null, current = null, showSourceSequence = false, onSelect, title: newTitle } = {}) {
    if (newTitle !== undefined) titleNode.textContent = newTitle;
    sourceSeq.setAttribute("display", showSourceSequence ? "inline" : "none");
    dynamic.replaceChildren();
    const visible = records.filter(
      (r) => (filters.phase === "all" || r.phase === filters.phase) && (filters.group === "all" || r.trueGroup === filters.group),
    );
    for (const r of visible) {
      const isCurrent = current && current.trialNumber === r.trialNumber;
      if (isCurrent) continue; // drawn last, on top
      const dot = el("circle", {
        cx: sx(r.pc1),
        cy: sy(r.pc2),
        r: S.presentedRadius[r.phase] ?? 3.6,
        fill: S.groupColor[r.trueGroup],
        stroke: r.phase === "training" ? "white" : "none",
        "stroke-width": r.phase === "training" ? 0.8 : 0,
        "data-trial": r.trialNumber,
        style: "cursor:pointer",
      });
      dot.append(
        el(
          "title",
          {},
          `Trial ${r.trialNumber} (${r.phase} #${r.phaseTrialNumber}) · group ${r.trueGroup} · ${r.imageId} · ${Number(r.stimulusAngle).toFixed(2)}° · response "${r.response}"`,
        ),
      );
      if (onSelect) dot.addEventListener("click", () => onSelect(r));
      dynamic.append(dot);
    }
    if (centers) {
      const radius = Math.hypot(centers.centerA[0], centers.centerA[1]);
      centerCircle.setAttribute("cx", sx(0));
      centerCircle.setAttribute("cy", sy(0));
      centerCircle.setAttribute("r", radius * unit);
      const [ax, ay] = [sx(centers.centerA[0]), sy(centers.centerA[1])];
      dynamic.append(el("path", { d: `M${ax - 7},${ay - 7}L${ax + 7},${ay + 7}M${ax + 7},${ay - 7}L${ax - 7},${ay + 7}`, stroke: "black", "stroke-width": 2.5 }));
      dynamic.append(el("text", { x: ax + 9, y: ay - 8, "font-weight": "bold" }, "A"));
      const [bx, by] = [sx(centers.centerB[0]), sy(centers.centerB[1])];
      dynamic.append(el("path", { d: starPath(bx, by, 9), fill: "black" }));
      dynamic.append(el("text", { x: bx + 9, y: by - 8, "font-weight": "bold" }, "B"));
    }
    if (current) {
      const passes = (filters.phase === "all" || current.phase === filters.phase) && (filters.group === "all" || current.trueGroup === filters.group);
      if (passes) {
        dynamic.append(el("circle", { cx: sx(current.pc1), cy: sy(current.pc2), r: 4.5, fill: S.groupColor[current.trueGroup] }));
        dynamic.append(
          el("circle", { cx: sx(current.pc1), cy: sy(current.pc2), r: S.current.radius, fill: "none", stroke: S.current.stroke, "stroke-width": S.current.width }),
        );
      }
    }
    return visible.length;
  }

  return { svg, update };
}
