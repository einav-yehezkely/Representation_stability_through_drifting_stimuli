// Trials in presentation order, each with its three faces: the sample, then the left and right
// choices. Used for a session's actual image order and for the planned order of the seed. Each
// face carries its group as a letter as well as a color, so group identity never depends on color
// alone.

import { h } from "../components/dom.js";

const faceUrl = (facesDirectory, imageId) => `${facesDirectory}${encodeURIComponent(imageId)}`;
const otherGroup = (group) => (group === "A" ? "B" : "A");

/**
 * item: { trialNumber, sampleGroup, sampleImageId, optionImageIds: {A, B}, leftGroup,
 *         chosenGroup?, correct? (true | false | undefined when not answered), planned?, record? }
 */
export function createImageGrid({ facesDirectory, onSelect }) {
  const element = h("div", { class: "image-grid" });
  let selectedButton = null;

  const face = (imageId, group, chosen = false) =>
    h(
      "span",
      { class: `trial-face${chosen ? " is-chosen" : ""}`, "data-group": group },
      h("img", { src: faceUrl(facesDirectory, imageId), alt: "", loading: "lazy", decoding: "async", width: 48, height: 59 }),
      h("span", { class: "trial-face-group", text: group }),
    );

  function thumb(item) {
    const answered = item.correct === true || item.correct === false;
    const rightGroup = otherGroup(item.leftGroup);
    const button = h(
      "button",
      {
        type: "button",
        class: `trial-thumb${answered ? (item.correct ? " is-correct" : " is-incorrect") : ""}`,
        title: `Trial ${item.trialNumber} · sample from ${item.sampleGroup}${answered ? ` · chose ${item.chosenGroup}` : ""}`,
        onclick: () => {
          selectedButton?.classList.remove("selected");
          selectedButton = button;
          button.classList.add("selected");
          onSelect?.(item);
        },
      },
      h("span", { class: "trial-thumb-head" }, h("span", { class: "trial-thumb-num", text: item.trialNumber }), answered ? h("span", { class: "trial-thumb-answer", text: item.correct ? "✓" : "✗" }) : null),
      h(
        "span",
        { class: "trial-faces" },
        face(item.sampleImageId, item.sampleGroup),
        h("span", { class: "trial-faces-gap", "aria-hidden": "true", text: "→" }),
        face(item.optionImageIds[item.leftGroup], item.leftGroup, item.chosenGroup === item.leftGroup),
        face(item.optionImageIds[rightGroup], rightGroup, item.chosenGroup === rightGroup),
      ),
    );
    return button;
  }

  return {
    element,
    /** @param {object[]} items  @param {{empty?: string}} options */
    setItems(items, { empty = "No trials." } = {}) {
      selectedButton = null;
      if (!items.length) {
        element.replaceChildren(h("p", { class: "muted", text: empty }));
        return;
      }
      element.replaceChildren(
        h(
          "div",
          { class: "image-grid-section" },
          h("h3", { class: "image-grid-title" }, "Trials", h("span", { class: "muted", text: ` · ${items.length}` })),
          h("div", { class: "trial-thumbs" }, items.map(thumb)),
        ),
      );
    },
  };
}

/** Large view of the three faces of one trial with its details (shown next to the grid). */
export function imageDetails(facesDirectory, item, rows) {
  const large = (imageId, group, label) =>
    h("figure", { class: "trial-detail-face" }, h("img", { src: faceUrl(facesDirectory, imageId), alt: `${label}: ${imageId}`, width: 89, height: 109, "data-group": group }), h("figcaption", { class: "muted small", text: label }));
  return h(
    "div",
    { class: "image-details" },
    h(
      "div",
      { class: "trial-detail-faces" },
      large(item.sampleImageId, item.sampleGroup, `Sample (${item.sampleGroup})`),
      large(item.optionImageIds.A, "A", "Option A"),
      large(item.optionImageIds.B, "B", "Option B"),
    ),
    h(
      "dl",
      {},
      rows.filter(([, v]) => v !== null && v !== undefined).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: String(v) })]),
    ),
  );
}
