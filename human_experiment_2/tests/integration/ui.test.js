// @vitest-environment jsdom
// Participant UI in a DOM: consent, instructions, a full run, failures, completion.
import { describe, expect, it } from "vitest";
import { mountExperiment } from "../../src/app.js";
import { completionScreen, consentScreen } from "../../src/components/screens.js";
import { TrialScreen } from "../../src/components/trialScreen.js";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { resolveEffectiveConfig } from "../../src/config/validateConfig.js";
import { consentContent } from "../../src/content/consent.js";
import { instructionsContent } from "../../src/content/instructions.js";
import { uiText } from "../../src/content/uiText.js";
import { ExperimentEngine, SCREENS } from "../../src/experiment/experimentEngine.js";
import { MemoryStorage } from "../../src/services/dataStorage/memoryStorage.js";
import { fakeClock, realPca } from "../helpers/engineHarness.js";

function setup({ developmentMode = false, overrides = { "drift.numberOfTrials": 3 }, loadDelay = 0, failLoads = 0 } = {}) {
  document.body.innerHTML = '<main id="app"></main>';
  const root = document.getElementById("app");
  let now = 0;
  let failures = failLoads;
  const config = resolveEffectiveConfig(experimentConfig, overrides);
  const storage = new MemoryStorage();
  const engine = new ExperimentEngine({
    config,
    developmentMode,
    pca: realPca(),
    storage,
    presenter: null,
    clock: fakeClock(),
    sessionId: "ui-session",
    sessionSeed: 1,
    identity: { participantId: "P", participantIdSource: "prolific_url" },
  });
  const trialScreen = new TrialScreen({
    now: () => now,
    raf: (cb) => setTimeout(cb, 0),
    loadImage: async (url) => {
      now += loadDelay;
      if (failures > 0) {
        failures -= 1;
        throw new Error("404");
      }
      const img = document.createElement("img");
      img.src = url;
      return img;
    },
    onResponse: (response, time) => engine.respond(response, time),
  });
  engine.presenter = trialScreen;
  mountExperiment(root, engine, trialScreen, { developmentMode });
  const tick = () => new Promise((r) => setTimeout(r, 5));
  const advance = (ms) => {
    now += ms;
  };
  return { root, engine, trialScreen, storage, tick, advance, config };
}

const faces = (root) => [...root.querySelectorAll("button.choice")];
const shownImages = (root) => [...root.querySelectorAll("img")].map((img) => img.getAttribute("src"));
const click = (el) => el.dispatchEvent(new MouseEvent("click", { bubbles: true }));

async function waitFor(predicate, tick) {
  for (let i = 0; i < 200; i += 1) {
    if (predicate()) return;
    await tick();
  }
  throw new Error("condition not reached");
}

describe("consent", () => {
  it("Continue stays disabled until the checkbox is checked; exact checkbox wording", () => {
    let accepted = 0;
    const el = consentScreen({ onAccept: () => (accepted += 1), onBypass: null });
    document.body.replaceChildren(el);
    const continueButton = [...el.querySelectorAll("button")].find((b) => b.textContent === "Continue");
    expect(el.textContent).toContain("I have read the information above and agree to participate in this study.");
    expect(continueButton.disabled).toBe(true);
    const box = el.querySelector("input[type=checkbox]");
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    expect(continueButton.disabled).toBe(false);
    click(continueButton);
    expect(accepted).toBe(1);
    expect(el.querySelector("[data-dev]")).toBeNull();
  });

  it("consent text contains researcher placeholders, not invented approvals", () => {
    const text = JSON.stringify(consentContent);
    expect(text).toContain("[RESEARCHER:");
    expect(text).not.toMatch(/approved by|IRB #\d|protocol \d/i);
  });

  it("participant mode has no bypass control; development mode does", () => {
    const participant = setup();
    expect(participant.root.querySelector('[data-dev="bypass-consent"]')).toBeNull();
    const dev = setup({ developmentMode: true });
    expect(dev.root.querySelector('[data-dev="bypass-consent"]')).not.toBeNull();
  });
});

describe("instructions", () => {
  it("explain the task without disclosing groups or the drift", () => {
    const text = JSON.stringify(instructionsContent).toLowerCase();
    expect(text).toContain("more similar");
    expect(text).not.toMatch(/drift|move|moving|rotat|shift|chang|group/);
  });
});

describe("trial screen", () => {
  it("full flow: sample, blank, two faces, no feedback, completion", async () => {
    const { root, engine, tick, advance } = setup();
    const box = root.querySelector("#consent-checkbox");
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Continue"));
    await waitFor(() => engine.state.screen === SCREENS.INSTRUCTIONS, tick);
    expect(root.textContent).toContain(instructionsContent.paragraphs[2]);

    const steps = [];
    engine.subscribe((st) => {
      if (st.status === "sample") steps.push(["sample", shownImages(root).length]);
      if (st.status === "blank") steps.push(["blank", shownImages(root).length]);
      if (st.status === "responding") steps.push(["responding", shownImages(root).length]);
    });
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Start"));

    const sides = ["left", "right", "left"];
    for (let i = 0; i < 3; i += 1) {
      await waitFor(() => engine.state.status === "responding", tick);
      expect(root.textContent).toContain(uiText.trial.prompt);
      expect(faces(root).every((f) => !f.disabled)).toBe(true);
      advance(300);
      click(root.querySelector(`button[data-response="${sides[i]}"]`));
      expect(faces(root).every((f) => f.disabled)).toBe(true);
      expect(shownImages(root)).toEqual([]); // the screen is empty from the click on
      await waitFor(() => engine.records.length === i + 1, tick);
      expect(root.textContent).not.toMatch(/Correct|Incorrect|Group A|Group B/);
    }
    await waitFor(() => engine.state.screen === SCREENS.COMPLETE, tick);
    expect(steps).toEqual(Array(3).fill([["sample", 1], ["blank", 0], ["responding", 2]]).flat());
    expect(engine.records.map((r) => r.response)).toEqual(sides);
    expect(root.textContent).toContain(uiText.completion.savedMessage);
    expect(root.querySelectorAll("button, a")).toHaveLength(0); // no data download for participants
  });

  it("response time excludes load time; a double click is ignored", async () => {
    const { root, engine, tick, advance } = setup({ loadDelay: 1500 });
    await engine.acceptConsent();
    await engine.startExperiment();
    await waitFor(() => engine.state.status === "responding", tick);
    advance(321);
    const [left, right] = faces(root);
    click(left);
    click(left);
    click(right);
    await waitFor(() => engine.records.length === 1, tick);
    await tick();
    expect(engine.records).toHaveLength(1);
    expect(engine.records[0]).toMatchObject({ responseTimeMs: 321, response: "left" });
  });

  it("an image load failure shows an English error with retry and creates no record", async () => {
    const { root, engine, tick } = setup({ failLoads: 1 });
    await engine.acceptConsent();
    await engine.startExperiment();
    await waitFor(() => engine.state.screen === SCREENS.ERROR, tick);
    expect(root.textContent).toContain("The picture could not be loaded");
    expect(engine.records).toHaveLength(0);
    click([...root.querySelectorAll("button")].find((x) => x.textContent === "Try again"));
    await waitFor(() => engine.state.status === "responding", tick);
    expect(faces(root).every((f) => !f.disabled)).toBe(true);
    expect(shownImages(root)).toHaveLength(2);
  });
});

describe("completion screen", () => {
  const url = "https://app.prolific.com/submissions/complete?cc=TEST123";

  it("online collection: confirms the save, links back to Prolific and offers no download", () => {
    const el = completionScreen({ state: { completionStatus: "complete", savedToServer: true, exported: true }, completionUrl: url });
    expect(el.textContent).toContain(uiText.completion.savedMessage);
    expect(el.querySelector("a.button").getAttribute("href")).toBe(url);
    expect(el.querySelector("a.button").textContent).toBe(uiText.completion.returnToProlificLabel);
    expect(el.querySelectorAll("button")).toHaveLength(0);
  });

  it("online collection without a configured link shows no link", () => {
    const el = completionScreen({ state: { completionStatus: "complete", savedToServer: true }, completionUrl: "" });
    expect(el.querySelector("a")).toBeNull();
  });

  it("participants are never offered a data download (browser storage or server)", () => {
    for (const state of [{ completionStatus: "complete" }, { completionStatus: "complete", savedToServer: true }, { completionStatus: "incomplete_dev_early_finish" }]) {
      const el = completionScreen({ state, completionUrl: "" });
      expect(el.querySelectorAll("button")).toHaveLength(0);
      expect(el.textContent).not.toMatch(/download/i);
      expect(el.textContent).toContain(uiText.completion.savedMessage);
    }
  });
});
