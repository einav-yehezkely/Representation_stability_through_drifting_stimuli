// Thumbnails of faces in presentation order, split by phase. Used for a session's actual image
// order and for the planned order of a seed. Each thumbnail carries its group as a letter as well
// as a color, so group identity never depends on color alone.

import { h } from "../components/dom.js";
import { RESPONSE_SHORT } from "./format.js";

const PHASE_TITLES = { training: "Training", drift: "Drift" };

/**
 * item: { trialNumber, phase, phaseTrialNumber, group, imageId, stimulusAngle, angleA,
 *         response?, correct? (true | false | null when not answered), planned? }
 */
export function createImageGrid({ facesDirectory, onSelect }) {
  const element = h("div", { class: "image-grid" });
  let selectedButton = null;

  function thumb(item) {
    const answered = item.correct === true || item.correct === false;
    const button = h(
      "button",
      {
        type: "button",
        class: `thumb${answered ? (item.correct ? " is-correct" : " is-incorrect") : ""}`,
        "data-group": item.group,
        title: `#${item.trialNumber} · ${item.phase} ${item.phaseTrialNumber} · group ${item.group} · ${item.imageId}${item.response ? ` · answer "${item.response}"` : ""}`,
        onclick: () => {
          selectedButton?.classList.remove("selected");
          selectedButton = button;
          button.classList.add("selected");
          onSelect?.(item);
        },
      },
      h("img", { src: `${facesDirectory}${encodeURIComponent(item.imageId)}`, alt: "", loading: "lazy", decoding: "async", width: 64, height: 78 }),
      h("span", { class: "thumb-num", text: item.phaseTrialNumber }), // numbered within its phase section
      h("span", { class: "thumb-group", text: item.group }),
      answered ? h("span", { class: "thumb-answer", text: `${item.correct ? "✓" : "✗"} ${RESPONSE_SHORT[item.response] ?? ""}` }) : null,
    );
    return button;
  }

  return {
    element,
    /** @param {object[]} items  @param {{empty?: string}} options */
    setItems(items, { empty = "No images." } = {}) {
      selectedButton = null;
      if (!items.length) {
        element.replaceChildren(h("p", { class: "muted", text: empty }));
        return;
      }
      const sections = ["training", "drift"]
        .map((phase) => [phase, items.filter((item) => item.phase === phase)])
        .filter(([, list]) => list.length)
        .map(([phase, list]) =>
          h(
            "div",
            { class: "image-grid-section" },
            h("h3", { class: "image-grid-title" }, PHASE_TITLES[phase], h("span", { class: "muted", text: ` · ${list.length} ${list.length === 1 ? "image" : "images"}` })),
            h("div", { class: "thumbs" }, list.map(thumb)),
          ),
        );
      element.replaceChildren(...sections);
    },
  };
}

/** Large view of one image with its details (shown next to the grid). */
export function imageDetails(facesDirectory, item, rows) {
  return h(
    "div",
    { class: "image-details" },
    h("img", { src: `${facesDirectory}${encodeURIComponent(item.imageId)}`, alt: item.imageId, width: 178, height: 218, "data-group": item.group }),
    h(
      "dl",
      {},
      rows.filter(([, v]) => v !== null && v !== undefined).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: String(v) })]),
    ),
  );
}
