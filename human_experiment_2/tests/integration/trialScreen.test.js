// @vitest-environment jsdom
// Trial screen: the three steps of a trial, preloading, answer echo; consent rendering.
import { describe, expect, it } from "vitest";
import { consentScreen } from "../../src/components/screens.js";
import { TrialScreen } from "../../src/components/trialScreen.js";
import { consentContent } from "../../src/content/consent.js";
import { ImageLoadError } from "../../src/experiment/experimentEngine.js";

function makeScreen({ failPreload = false } = {}) {
  let now = 0;
  const loads = [];
  const preloads = [];
  const responses = [];
  const image = (url) => Object.assign(document.createElement("img"), { src: url });
  const screen = new TrialScreen({
    now: () => now,
    raf: (cb) => setTimeout(cb, 0),
    loadImage: async (url) => {
      loads.push(url);
      return image(url);
    },
    preloadImage: async (url) => {
      preloads.push(url);
      if (failPreload) throw new Error("preload failed");
      return image(url);
    },
    onResponse: (side, time) => responses.push([side, time]),
  });
  document.body.replaceChildren(screen.element);
  return { screen, loads, preloads, responses, advance: (ms) => (now += ms) };
}

const images = (screen) => [...screen.element.querySelectorAll("img")].map((img) => img.getAttribute("src"));
const click = (el) => el.dispatchEvent(new MouseEvent("click", { bubbles: true }));

async function showTrial(screen, [sample, left, right]) {
  await screen.load([sample, left, right]);
  await screen.showSample(sample);
  await screen.hideSample();
  return screen.showPair(left, right);
}

describe("the steps of a trial", () => {
  it("shows the sample alone, then nothing, then two faces", async () => {
    const { screen } = makeScreen();
    await screen.load(["s.jpg", "l.jpg", "r.jpg"]);
    await screen.showSample("s.jpg");
    expect(images(screen)).toEqual(["s.jpg"]);
    expect(screen.pair.hidden).toBe(true);
    await screen.hideSample();
    expect(images(screen)).toEqual([]);
    expect(screen.sampleFrame.hidden).toBe(true);
    expect(screen.pair.hidden).toBe(true);
    await screen.showPair("l.jpg", "r.jpg");
    expect(images(screen)).toEqual(["l.jpg", "r.jpg"]);
    expect(screen.choices.map((c) => c.dataset.response)).toEqual(["left", "right"]);
    expect(screen.pair.hidden).toBe(false);
  });

  it("the faces are clickable only from the moment the two are painted, and the onset is that moment", async () => {
    const { screen, advance, responses } = makeScreen();
    await screen.load(["s.jpg", "l.jpg", "r.jpg"]);
    await screen.showSample("s.jpg");
    click(screen.choices[0]);
    await screen.hideSample();
    advance(1500);
    const shown = screen.showPair("l.jpg", "r.jpg");
    expect(screen.choices.every((c) => c.disabled)).toBe(true);
    expect(await shown).toBe(1500);
    expect(screen.choices.every((c) => !c.disabled)).toBe(true);
    expect(responses).toEqual([]);
  });

  it("registers one click with its time", async () => {
    const { screen, advance, responses } = makeScreen();
    await showTrial(screen, ["s.jpg", "l.jpg", "r.jpg"]);
    advance(321);
    click(screen.choices[1]);
    click(screen.choices[1]);
    click(screen.choices[0]);
    expect(responses).toEqual([["right", 321]]);
    expect(screen.choices.every((c) => c.disabled)).toBe(true);
  });

  it("the DOM carries nothing that reveals a group", async () => {
    const { screen } = makeScreen();
    await showTrial(screen, ["s.jpg", "l.jpg", "r.jpg"]);
    expect(screen.element.innerHTML).not.toMatch(/group|data-group/i);
    expect(screen.element.textContent).not.toMatch(/\b[AB]\b|Correct|Incorrect/);
  });
});

describe("loading and preloading", () => {
  it("uses preloaded images without loading them again; others are loaded on demand", async () => {
    const { screen, loads, preloads } = makeScreen();
    screen.preload(["a.jpg", "b.jpg"]);
    screen.preload(["a.jpg", "b.jpg"]); // already requested: not fetched twice
    expect(preloads).toEqual(["a.jpg", "b.jpg"]);
    await screen.load(["a.jpg", "b.jpg", "c.jpg"]);
    expect(loads).toEqual(["c.jpg"]);
  });

  it("drops images that are no longer announced", async () => {
    const { screen, loads } = makeScreen();
    screen.preload(["a.jpg"]);
    screen.preload(["b.jpg"]);
    await screen.load(["a.jpg"]);
    expect(loads).toEqual(["a.jpg"]);
  });

  it("a failed preload falls back to a normal load; a failing load raises ImageLoadError", async () => {
    const { screen, loads } = makeScreen({ failPreload: true });
    screen.preload(["a.jpg"]);
    await screen.load(["a.jpg"]);
    expect(loads).toEqual(["a.jpg"]);
    screen.loadImage = async () => {
      throw new Error("404");
    };
    await expect(screen.load(["x.jpg"])).rejects.toBeInstanceOf(ImageLoadError);
  });

  it("a retry after a failed load keeps the images already loaded", async () => {
    const { screen, loads } = makeScreen();
    const working = screen.loadImage;
    let fail = true;
    screen.loadImage = async (url) => {
      if (url === "bad.jpg" && fail) throw new Error("404");
      return working(url);
    };
    await expect(screen.load(["ok.jpg", "bad.jpg"])).rejects.toBeInstanceOf(ImageLoadError);
    fail = false;
    await screen.load(["ok.jpg", "bad.jpg"]);
    expect(loads.filter((url) => url === "ok.jpg")).toHaveLength(1);
  });
});

describe("consent screen", () => {
  it("renders every section heading and body as elements (not '[object HTML…]' text)", () => {
    const el = consentScreen({ onAccept: () => {}, onBypass: null });
    expect(el.textContent).not.toContain("[object");
    expect([...el.querySelectorAll(".consent-section h2")].map((n) => n.textContent)).toEqual(consentContent.sections.map((s) => s.heading));
  });
});
