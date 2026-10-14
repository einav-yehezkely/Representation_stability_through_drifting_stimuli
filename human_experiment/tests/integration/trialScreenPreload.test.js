// @vitest-environment jsdom
// Trial screen preloading, answer echo and feedback styling; consent rendering.
import { describe, expect, it } from "vitest";
import { consentScreen } from "../../src/components/screens.js";
import { TrialScreen } from "../../src/components/trialScreen.js";
import { consentContent } from "../../src/content/consent.js";
import { ImageLoadError } from "../../src/experiment/experimentEngine.js";

function makeScreen({ failPreload = false } = {}) {
  let now = 0;
  const loads = [];
  const preloads = [];
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
    onResponse: () => {},
  });
  document.body.replaceChildren(screen.element);
  return { screen, loads, preloads, advance: (ms) => (now += ms) };
}

describe("trial screen preloading", () => {
  it("shows a preloaded image without loading it again; others are loaded on demand", async () => {
    const { screen, loads, preloads } = makeScreen();
    screen.preload(["a.jpg", "b.jpg"]);
    screen.preload(["a.jpg", "b.jpg"]); // already requested: not fetched twice
    expect(preloads).toEqual(["a.jpg", "b.jpg"]);
    await screen.show("a.jpg");
    expect(loads).toEqual([]);
    expect(screen.element.querySelectorAll("img")).toHaveLength(1);
    expect(screen.element.querySelector("img").getAttribute("src")).toBe("a.jpg");
    await screen.show("c.jpg");
    expect(loads).toEqual(["c.jpg"]);
  });

  it("drops images that are no longer announced", async () => {
    const { screen, loads } = makeScreen();
    screen.preload(["a.jpg"]);
    screen.preload(["b.jpg"]);
    await screen.show("a.jpg");
    expect(loads).toEqual(["a.jpg"]);
  });

  it("a failed preload falls back to a normal load (a failing load still raises ImageLoadError)", async () => {
    const { screen, loads } = makeScreen({ failPreload: true });
    screen.preload(["a.jpg"]);
    await screen.show("a.jpg");
    expect(loads).toEqual(["a.jpg"]);
    screen.loadImage = async () => {
      throw new Error("404");
    };
    screen.preload(["x.jpg"]);
    await expect(screen.show("x.jpg")).rejects.toBeInstanceOf(ImageLoadError);
  });

  it("onset is taken after the preloaded image is painted, and buttons enable at that moment", async () => {
    const { screen, advance } = makeScreen();
    screen.preload(["a.jpg"]);
    advance(500); // time passes before the trial starts
    const shown = screen.show("a.jpg");
    expect(screen.buttons.every((b) => b.disabled)).toBe(true);
    const onset = await shown;
    expect(onset).toBe(500);
    expect(screen.buttons.every((b) => !b.disabled)).toBe(true);
  });

  it("marks the given answer while saving and resets it for the next face", async () => {
    const { screen } = makeScreen();
    await screen.show("a.jpg");
    screen.buttons[2].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    expect(screen.buttons.map((b) => b.classList.contains("selected"))).toEqual([false, false, true, false]);
    expect(screen.buttons.every((b) => b.disabled)).toBe(true);
    await screen.show("b.jpg");
    expect(screen.buttons.some((b) => b.classList.contains("selected"))).toBe(false);
  });

  it("feedback carries its outcome only while text is shown", () => {
    const { screen } = makeScreen();
    screen.setFeedback("Correct", true);
    expect(screen.feedback.dataset.outcome).toBe("correct");
    expect(screen.feedback.textContent).toBe("Correct");
    screen.setFeedback("Incorrect", false);
    expect(screen.feedback.dataset.outcome).toBe("incorrect");
    screen.setFeedback(null);
    expect(screen.feedback.dataset.outcome).toBeUndefined();
    expect(screen.feedback.textContent).toBe("");
  });
});

describe("consent screen", () => {
  it("renders every section heading and body as elements (not '[object HTML…]' text)", () => {
    const el = consentScreen({ onAccept: () => {}, onBypass: null });
    expect(el.textContent).not.toContain("[object");
    expect([...el.querySelectorAll(".consent-section h2")].map((n) => n.textContent)).toEqual(consentContent.sections.map((s) => s.heading));
  });
});
