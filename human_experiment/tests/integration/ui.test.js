// @vitest-environment jsdom
// Participant UI in a DOM (acceptance checks 3, 5, 7, 12, 13).
import { describe, expect, it } from "vitest";
import { mountExperiment } from "../../src/app.js";
import { consentScreen } from "../../src/components/screens.js";
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

const buttons = (root) => [...root.querySelectorAll(".response-buttons button")];
const click = (el) => el.dispatchEvent(new MouseEvent("click", { bubbles: true }));

async function waitFor(predicate, tick) {
  for (let i = 0; i < 200; i += 1) {
    if (predicate()) return;
    await tick();
  }
  throw new Error("condition not reached");
}

describe("consent (PRD §4)", () => {
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

describe("instructions and transition wording", () => {
  it("instructions explain the four responses and feedback, without disclosing drift", () => {
    const text = JSON.stringify(instructionsContent).toLowerCase();
    for (const r of ["\"a\"", "probably a", "probably b", "\"b\""]) expect(text).toContain(r);
    expect(text).toContain("feedback");
    expect(text).not.toMatch(/drift|move|moving|rotat|shift|chang/);
  });

  it("transition text is exact, announces the end of feedback and not category movement", () => {
    const t = uiText.transition;
    expect(t.heading).toBe("The first part of the experiment is complete.");
    expect(t.paragraphs).toEqual([
      "In the next part, you will continue performing the same task, but you will no longer receive feedback about your answers.",
      "Please continue to classify each face as accurately as possible.",
    ]);
    expect(t.continueLabel).toBe("Continue");
    expect(JSON.stringify(t).toLowerCase()).not.toMatch(/drift|move|rotat|chang/);
  });
});

describe("trial screen (acceptance checks 3, 5, 7, 8)", () => {
  it("full flow: one face, fixed buttons, training feedback, transition, drift without feedback", async () => {
    const { root, engine, tick, advance } = setup();
    // Consent
    const box = root.querySelector("#consent-checkbox");
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Continue"));
    await waitFor(() => engine.state.screen === SCREENS.INSTRUCTIONS, tick);
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Start"));

    // Training: 20 correct answers. Record the feedback text rendered in the DOM per state.
    const feedbackSeen = [];
    engine.subscribe((st) => {
      if (st.status === "feedback") feedbackSeen.push(root.querySelector(".feedback")?.textContent ?? null);
    });
    for (let i = 0; i < 20; i += 1) {
      await waitFor(() => engine.state.status === "responding", tick);
      expect(root.querySelectorAll("img")).toHaveLength(1);
      expect(buttons(root).map((b) => b.textContent)).toEqual(["A", "Probably A", "Probably B", "B"]);
      expect(buttons(root).every((b) => !b.disabled)).toBe(true);
      advance(400);
      const group = engine.pending.group;
      click(buttons(root).find((b) => b.textContent === group));
      expect(buttons(root).every((b) => b.disabled)).toBe(true);
      await waitFor(() => engine.records.length === i + 1 && (engine.state.status === "responding" || engine.state.screen === SCREENS.TRANSITION), tick);
    }
    expect(feedbackSeen).toEqual(Array(20).fill("Correct"));
    expect(engine.records.every((r) => r.feedbackShown)).toBe(true);
    await waitFor(() => engine.state.screen === SCREENS.TRANSITION, tick);
    expect(root.textContent).toContain("you will no longer receive feedback");
    expect(root.querySelectorAll("img")).toHaveLength(0);

    // Drift waits for Continue.
    await tick();
    expect(engine.records).toHaveLength(20);
    click([...root.querySelectorAll("button")].find((b) => b.textContent === "Continue"));
    for (let i = 0; i < 3; i += 1) {
      await waitFor(() => engine.state.status === "responding", tick);
      expect(root.querySelectorAll("img")).toHaveLength(1);
      expect(root.querySelector(".feedback").textContent).toBe("");
      advance(300);
      click(buttons(root)[i]);
      await tick();
      expect(root.querySelector(".feedback")?.textContent ?? "").toBe("");
      expect(root.textContent).not.toMatch(/Correct|Incorrect|Group A|Group B/);
    }
    await waitFor(() => engine.state.screen === SCREENS.COMPLETE, tick);
    expect(engine.driftRecords.map((r) => r.response)).toEqual(["A", "Probably A", "Probably B"]);
    expect(root.textContent).toContain("Download data");
  });

  it("buttons stay disabled while the image loads; RT excludes load time; double click is ignored", async () => {
    const { root, engine, tick, advance } = setup({ loadDelay: 1500 });
    await engine.acceptConsent();
    await engine.startTraining();
    await waitFor(() => engine.state.status === "responding", tick);
    advance(321);
    const b = buttons(root)[0];
    click(b);
    click(b);
    click(buttons(root)[3]);
    await waitFor(() => engine.records.length === 1, tick);
    await tick();
    expect(engine.records).toHaveLength(1);
    expect(engine.records[0].responseTimeMs).toBe(321);
  });

  it("an image load failure shows an English error with retry and creates no record", async () => {
    const { root, engine, tick } = setup({ failLoads: 1 });
    await engine.acceptConsent();
    await engine.startTraining();
    await waitFor(() => engine.state.screen === SCREENS.ERROR, tick);
    expect(root.textContent).toContain("The picture could not be loaded");
    expect(engine.records).toHaveLength(0);
    click([...root.querySelectorAll("button")].find((x) => x.textContent === "Try again"));
    await waitFor(() => engine.state.status === "responding", tick);
    expect(root.querySelectorAll("img")).toHaveLength(1);
  });
});
