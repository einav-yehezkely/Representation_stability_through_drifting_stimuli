// Trial screen and stimulus presenter.
//
// A trial is shown in three steps: the sample face alone in the center, an empty screen, then two
// faces side by side. The faces are the response controls: the participant clicks one of them, and
// the engine clears the screen at once. The
// screen never shows group labels, colors or outcome cues, and the DOM does not carry the group.
//
// Onset procedure (render/paint-aware):
//   1. All three images of the trial are fetched and decoded off-screen (load) before anything is
//      shown, so loading never affects the sample duration, the blank or the response time. Images
//      the engine announced in advance (preload) are already decoded.
//   2. Each step changes the DOM and then waits for requestAnimationFrame -> requestAnimationFrame:
//      the second callback runs after the frame with the change has been rendered, and the time of
//      the step is taken there.
//   3. For the two choices, onset = performance.now() is taken at that moment and the faces become
//      clickable at the same instant.
//   4. On a click, performance.now() is captured first, before any other work, and passed to the
//      engine, which computes responseTime = click - onset.

import { uiText } from "../content/uiText.js";
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
    this.ready = new Map(); // url -> decoded image of the current trial
    // Fixed size (aspect ratio of the source images), so the layout never moves between trials.
    const size = { width: `min(${SOURCE_WIDTH * scale}px, 44vw, calc(56vh * ${SOURCE_WIDTH / SOURCE_HEIGHT}))`, aspectRatio: `${SOURCE_WIDTH} / ${SOURCE_HEIGHT}` };
    this.sampleFrame = h("div", { class: "stimulus-frame", "data-role": "sample", hidden: true, style: size });
    this.choices = RESPONSES.map((side) =>
      h("button", { type: "button", class: "stimulus-frame choice", "data-response": side, "aria-label": `Choose the ${side} face`, disabled: true, style: size, onclick: (event) => this.handleClick(event, side) }),
    );
    this.pair = h("div", { class: "pair", hidden: true }, h("div", { class: "choices" }, this.choices), h("p", { class: "trial-prompt", text: uiText.trial.prompt }));
    this.element = h("div", { class: "trial" }, this.sampleFrame, this.pair);
  }

  static async decodeImage(url) {
    const img = new Image();
    img.alt = "";
    img.draggable = false;
    img.src = url;
    await img.decode();
    return img;
  }

  handleClick(event, side) {
    const time = this.now(); // captured before anything else
    if (event.currentTarget.disabled) return;
    this.disableResponses();
    this.onResponse(side, time);
  }

  // ---- presenter interface used by the engine

  /** Download and decode the next images in advance; images not listed any more are dropped. */
  preload(urls) {
    if (!this.preloadImage) return;
    const wanted = new Set(urls);
    for (const url of this.preloaded.keys()) if (!wanted.has(url)) this.preloaded.delete(url);
    for (const url of urls) {
      if (!this.preloaded.has(url)) this.preloaded.set(url, this.preloadImage(url).catch(() => null)); // failure: load() fetches it again
    }
  }

  /** Have every image of a trial decoded. Rejects with ImageLoadError; a retry keeps those already loaded. */
  async load(urls) {
    await Promise.all(
      urls.map(async (url) => {
        if (this.ready.has(url)) return;
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
        this.ready.set(url, img);
      }),
    );
  }

  take(url) {
    const img = this.ready.get(url);
    this.ready.delete(url);
    return img;
  }

  async showSample(url) {
    this.disableResponses();
    this.clear();
    this.sampleFrame.replaceChildren(this.take(url));
    this.sampleFrame.hidden = false;
    await nextPaint(this.raf);
    return this.now();
  }

  async hideSample() {
    this.clear();
    await nextPaint(this.raf);
    return this.now();
  }

  async showPair(leftUrl, rightUrl) {
    this.choices[0].replaceChildren(this.take(leftUrl));
    this.choices[1].replaceChildren(this.take(rightUrl));
    this.pair.hidden = false;
    await nextPaint(this.raf);
    const onset = this.now();
    this.enableResponses();
    return onset;
  }

  disableResponses() {
    for (const choice of this.choices) choice.disabled = true;
  }

  enableResponses() {
    for (const choice of this.choices) choice.disabled = false;
  }

  clear() {
    this.sampleFrame.replaceChildren();
    this.sampleFrame.hidden = true;
    this.pair.hidden = true;
    for (const choice of this.choices) choice.replaceChildren();
  }
}
