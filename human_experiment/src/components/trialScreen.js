// Trial screen and stimulus presenter (PRD §4 trial screen, §9 timing).
//
// Exactly one face image, centered, with four response buttons below it in the fixed order
// A / Probably A / Probably B / B. The screen never shows group labels, colors or outcome cues
// during drift; feedback text is rendered only when the engine supplies it (training).
//
// Onset procedure (documented, render/paint-aware):
//   1. Responses are disabled and the previous face is removed from the DOM.
//   2. The next image is fetched and decoded off-screen (new Image(); await img.decode()).
//      No timer runs during loading.
//   3. The decoded image is inserted into the (empty) stimulus frame.
//   4. requestAnimationFrame -> requestAnimationFrame: the second callback runs after the frame
//      containing the image has been rendered. At that moment onset = performance.now() is taken
//      and the response buttons are enabled — the same instant.
//   5. On a click, performance.now() is captured first, before any other work, and passed to the
//      engine, which computes responseTime = click - onset.

import { ImageLoadError } from "../experiment/experimentEngine.js";
import { RESPONSES } from "../experiment/responseCoding.js";
import { h } from "./dom.js";

const SOURCE_WIDTH = 178; // female_faces images are 178 x 218 px
const SOURCE_HEIGHT = 218;

function nextPaint(raf) {
  return new Promise((resolve) => raf(() => raf(() => resolve())));
}

export class TrialScreen {
  /**
   * @param {object} options
   * @param {(response:string, time:number) => void} options.onResponse
   * @param {number} options.scale display scale for the source images
   */
  constructor({ onResponse, scale = 2, now = () => performance.now(), raf = (cb) => requestAnimationFrame(cb), loadImage }) {
    this.onResponse = onResponse;
    this.now = now;
    this.raf = raf;
    this.loadImage = loadImage ?? TrialScreen.decodeImage;
    this.frame = h("div", {
      class: "stimulus-frame",
      style: { width: `${SOURCE_WIDTH * scale}px`, height: `${SOURCE_HEIGHT * scale}px`, maxWidth: "90vw", maxHeight: "60vh" },
    });
    this.feedback = h("div", { class: "feedback", "aria-live": "polite" });
    this.buttons = RESPONSES.map((label) =>
      h("button", { type: "button", "data-response": label, disabled: true, onclick: (event) => this.handleClick(event, label) }, label),
    );
    this.element = h("div", { class: "trial" }, this.frame, h("div", { class: "response-buttons" }, this.buttons), this.feedback);
  }

  static async decodeImage(url) {
    const img = new Image();
    img.alt = "";
    img.draggable = false;
    img.src = url;
    await img.decode();
    return img;
  }

  handleClick(event, label) {
    const time = this.now(); // captured before anything else
    if (event.currentTarget.disabled) return;
    this.disableResponses();
    this.onResponse(label, time);
  }

  // ---- presenter interface used by the engine

  async show(url) {
    this.disableResponses();
    this.clear();
    let img;
    try {
      img = await this.loadImage(url);
    } catch (error) {
      throw new ImageLoadError(url, error);
    }
    this.frame.replaceChildren(img);
    await nextPaint(this.raf);
    const onset = this.now();
    this.enableResponses();
    return onset;
  }

  disableResponses() {
    for (const button of this.buttons) button.disabled = true;
  }

  enableResponses() {
    for (const button of this.buttons) button.disabled = false;
  }

  clear() {
    this.frame.replaceChildren();
    this.setFeedback(null);
  }

  // ---- view state

  setFeedback(text) {
    this.feedback.textContent = text ?? "";
  }
}
