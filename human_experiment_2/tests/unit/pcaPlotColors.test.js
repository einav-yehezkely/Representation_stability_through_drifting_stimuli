// @vitest-environment jsdom
// Samples in the PCA plot: colored by their group, or by the group the participant chose.
import { describe, expect, it } from "vitest";
import { createPcaPlot } from "../../src/visualization/pcaPlot.js";

const pca = { count: 1, pc1: [0.1], pc2: [0.1], maxRadius: 0.5, axes: ["PC1", "PC2"] };

const records = [
  { trialNumber: 1, sampleGroup: "A", chosenGroup: "A", samplePc1: 0.1, samplePc2: 0, sampleAngle: 0, sampleImageId: "1.jpg" },
  { trialNumber: 2, sampleGroup: "A", chosenGroup: "B", samplePc1: 0.2, samplePc2: 0, sampleAngle: 0, sampleImageId: "2.jpg" },
  { trialNumber: 3, sampleGroup: "B", chosenGroup: "A", samplePc1: -0.2, samplePc2: 0, sampleAngle: 180, sampleImageId: "3.jpg" },
];

function draw(colorBy, current = null) {
  const plot = createPcaPlot({ pca, colorBy });
  plot.update({ records, current });
  const fill = (trial) => plot.svg.querySelector(`[data-trial="${trial}"]`)?.getAttribute("fill");
  return { svg: plot.svg, fill };
}

describe("PCA plot colors", () => {
  it("colors by the chosen group in prediction mode, wrong answers included", () => {
    const { fill } = draw("prediction");
    expect([fill(1), fill(2), fill(3)]).toEqual(["blue", "red", "blue"]);
  });

  it("colors by the sample's group by default", () => {
    const plot = createPcaPlot({ pca });
    plot.update({ records });
    const fills = [1, 2, 3].map((t) => plot.svg.querySelector(`[data-trial="${t}"]`).getAttribute("fill"));
    expect(fills).toEqual(["blue", "blue", "red"]);
  });

  it("colors the current sample the same way as the other points", () => {
    const { svg } = draw("prediction", records[2]);
    const dots = [...svg.querySelectorAll('[data-layer="dynamic"] circle')];
    expect(dots.find((c) => c.getAttribute("r") === "4.5").getAttribute("fill")).toBe("blue");
  });

  it("names the legend after what the colors mean", () => {
    expect(draw("prediction").svg.textContent).toContain("Chose A");
    expect(draw("group").svg.textContent).toContain("Sample from A");
  });
});
