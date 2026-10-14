// Trial screen and stimulus presenter (PRD §4 trial screen, §9 timing).
//
// Exactly one face image, centered, with four response buttons below it in the fixed order
// A / Probably A / Probably B / B. The screen never shows group labels, colors or outcome cues
// during drift; feedback text is rendered only when the engine supplies it (training).
//
// Onset procedure (documented, render/paint-aware):
//   1. Responses are disabled and the previous face is removed from the DOM.
//   2. The next image is fetched and decoded off-screen (new Image(); await img.decode()).
//      No timer runs during loading. When the engine announced the image in advance (preload),
//      it is already decoded and this step completes without waiting.
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
  constructor({ onResponse, scale = 2, now = () => performance.now(), raf = (cb) => requestAnimationFrame(cb), loadImage, preloadImage }) {
    this.onResponse = onResponse;
    this.now = now;
    this.raf = raf;
    this.loadImage = loadImage ?? TrialScreen.decodeImage;
    // A custom loader (tests) disables preloading unless a preloader is passed explicitly.
    this.preloadImage = preloadImage !== undefined ? preloadImage : loadImage ? null : TrialScreen.decodeImage;
    this.preloaded = new Map(); // url -> Promise<HTMLImageElement | null>
    this.frame = h("div", {
      class: "stimulus-frame",
      // Fixed size (aspect ratio of the source images), so the layout never moves between trials.
      style: { width: `min(${SOURCE_WIDTH * scale}px, 88vw, calc(56vh * ${SOURCE_WIDTH / SOURCE_HEIGHT}))`, aspectRatio: `${SOURCE_WIDTH} / ${SOURCE_HEIGHT}` },
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
    const button = event.currentTarget;
    if (button.disabled) return;
    this.disableResponses();
    button.classList.add("selected"); // shows which answer was registered
    this.onResponse(label, time);
  }

  // ---- presenter interface used by the engine

  /** Download and decode the next images in advance; images not listed any more are dropped. */
  preload(urls) {
    if (!this.preloadImage) return;
    const wanted = new Set(urls);
    for (const url of this.preloaded.keys()) if (!wanted.has(url)) this.preloaded.delete(url);
    for (const url of urls) {
      if (!this.preloaded.has(url)) this.preloaded.set(url, this.preloadImage(url).catch(() => null)); // failure: show() loads it again
    }
  }

  async show(url) {
    this.disableResponses();
    this.clear();
    const preloaded = this.preloaded.get(url);
    this.preloaded.delete(url);
    let img = preloaded ? await preloaded : null;
    if (!img) {
      try {
        img = await this.loadImage(url);
      } catch (error) {
        throw new ImageLoadError(url, error);
      }
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
    for (const button of this.buttons) {
      button.classList.remove("selected");
      button.disabled = false;
    }
  }

  clear() {
    this.frame.replaceChildren();
    this.setFeedback(null);
  }

  // ---- view state

  /** Training feedback only. `correct` selects the color; drift never calls this with text. */
  setFeedback(text, correct = null) {
    this.feedback.textContent = text ?? "";
    if (text && correct !== null) this.feedback.dataset.outcome = correct ? "correct" : "incorrect";
    else delete this.feedback.dataset.outcome;
  }
}
