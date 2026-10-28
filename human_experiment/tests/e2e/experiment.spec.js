// Real-browser smoke tests (Chrome). Run with `npm run build && npm run test:e2e`.
import fs from "node:fs";
import { expect, test } from "@playwright/test";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { parseCsv } from "../../src/services/dataStorage/csvSerializer.js";

const PROD = "http://localhost:4173";
const DEV = "http://localhost:5173";

async function giveConsent(page) {
  await expect(page.locator('[data-screen="consent"]')).toBeVisible();
  const cont = page.getByRole("button", { name: "Continue" });
  await expect(cont).toBeDisabled();
  await page.locator("#consent-checkbox").check();
  await expect(cont).toBeEnabled();
  await cont.click();
  await page.getByRole("button", { name: "Start" }).click();
}

async function storedSession(page) {
  return page.evaluate(() => {
    const index = JSON.parse(localStorage.getItem("hx:index") ?? "[]");
    const id = index.at(-1);
    return {
      meta: JSON.parse(localStorage.getItem(`hx:session:${id}:meta`)),
      trials: JSON.parse(localStorage.getItem(`hx:session:${id}:trials`) ?? "[]"),
    };
  });
}

/** Stand-in for save.php (vite preview cannot run PHP): records every request and acknowledges it. */
async function fakeSaveEndpoint(page) {
  const requests = [];
  // Experiment settings saved on the admin page: none, so the defaults apply.
  await page.route("**/save.php?settings=1", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, settings: null }) }),
  );
  await page.route("**/save.php", async (route) => {
    requests.push(JSON.parse(route.request().postData()));
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true }) });
  });
  return requests;
}

test("participant mode (production build): no dev controls, one face, feedback, server saving, reload", async ({ page }) => {
  const requests = await fakeSaveEndpoint(page);
  // ?dev=1 must be ignored by a production build.
  await page.goto(`${PROD}/index.html?PROLIFIC_PID=E2E-PID&dev=1`);
  await expect(page.locator("[data-dev]")).toHaveCount(0);
  await giveConsent(page);

  for (let i = 0; i < 5; i += 1) {
    const buttons = page.locator(".response-buttons button");
    await expect(buttons.first()).toBeEnabled();
    await expect(page.locator(".stimulus-frame img")).toHaveCount(1);
    await expect(page.locator("img")).toHaveCount(1);
    expect(await page.locator(".stimulus-frame img").evaluate((img) => img.complete && img.naturalWidth)).toBe(178);
    await expect(buttons).toHaveText(["A", "Probably A", "Probably B", "B"]);
    await buttons.nth(i % 4).click();
    await expect(page.locator(".feedback")).toHaveText(/^(Correct|Incorrect)$/);
  }
  await expect(page.locator(".dev-panel")).toHaveCount(0);

  // Every answer was sent to the server.
  const [startRequest, ...trialRequests] = requests;
  expect(startRequest.action).toBe("start");
  expect(startRequest.session).toMatchObject({ participantId: "E2E-PID", participantIdSource: "prolific_url", consentGiven: true, developmentMode: false, sessionSeed: experimentConfig.randomization.fixedSeed });
  expect(trialRequests.length).toBeGreaterThanOrEqual(4);
  for (const { action, trial } of trialRequests) {
    expect(action).toBe("trial");
    expect(trial.sessionId).toBe(startRequest.session.sessionId);
    expect(trial.responseTimeMs).toBeGreaterThan(0);
    expect(trial.responseTimeMs).toBeLessThan(5000);
    expect(trial.phase).toBe("training");
    expect(trial.feedbackShown).toBe(true);
  }
  // A local backup is kept as well.
  const { meta, trials } = await storedSession(page);
  expect(meta.sessionId).toBe(startRequest.session.sessionId);
  expect(trials.length).toBe(trialRequests.length);

  // Reload: no download is offered; the old session is marked interrupted and a new one starts.
  await page.reload();
  await expect(page.locator('[data-screen="consent"]')).toBeVisible();
  await expect(page.locator('[data-screen="recovery"]')).toHaveCount(0);
  expect(requests.at(-1)).toEqual({ action: "interrupted", sessionId: meta.sessionId });
});

test("development mode: overrides, bypass, full run to CSV, live PCA plot, review page", async ({ page, context }) => {
  await page.goto(`${DEV}/index.html?dev=1`);
  await expect(page.locator('[data-screen="dev-setup"]')).toBeVisible();
  await page.locator('[data-field="training.minTrials"]').fill("4");
  await page.locator('[data-field="training.accuracyWindow"]').fill("4");
  await page.locator('[data-field="drift.numberOfTrials"]').fill("6");
  await page.locator('[data-field="drift.degreesPerTrial"]').fill("30");
  await expect(page.locator('[data-field="seed"]')).toHaveCount(0); // the seed is fixed, not editable
  await page.getByRole("button", { name: "Start session" }).click();
  await page.locator('[data-dev="bypass-consent"]').click();
  await page.getByRole("button", { name: "Start" }).click();

  const groupOf = async () => (await page.locator(".dev-info dt:text-is('True group') + dd").textContent()).trim();
  const readyOrTransition = () =>
    page.waitForFunction(() => {
      if (document.querySelector('[data-screen="transition"]')) return true;
      const first = document.querySelector(".response-buttons button");
      return Boolean(first && !first.disabled);
    });

  // Training: answer correctly using the debug panel until the transition screen appears.
  let answered = 0;
  for (; answered < 40; answered += 1) {
    await readyOrTransition();
    if (await page.locator('[data-screen="transition"]').isVisible()) break;
    const group = await groupOf();
    await page.locator(`.response-buttons button[data-response="${group}"]`).click();
    await expect(page.locator(".feedback")).toHaveText("Correct");
  }
  expect(answered).toBe(4); // minTrials = window = 4, all correct
  await expect(page.locator('[data-screen="transition"]')).toBeVisible();

  // PCA plot in the dev panel.
  await page.getByRole("button", { name: "PCA plot" }).click();
  await expect(page.locator('.dev-plot [data-layer="all-points"] circle')).toHaveCount(11817);
  await expect(page.locator('.dev-plot [data-layer="dynamic"] circle[data-trial]')).toHaveCount(4);

  await page.getByRole("button", { name: "Continue" }).click();
  const downloads = [];
  page.on("download", (d) => downloads.push(d.suggestedFilename()));
  for (let i = 0; i < 6; i += 1) {
    const buttons = page.locator(".response-buttons button");
    await expect(buttons.first()).toBeEnabled();
    await expect(page.locator(".feedback")).toHaveText("");
    await buttons.nth(i % 4).click();
  }
  await expect(page.locator('[data-screen="complete"]')).toBeVisible();
  // Participants get no data download: nothing is downloaded and the screen offers no button.
  await page.waitForTimeout(500);
  expect(downloads).toEqual([]);
  await expect(page.locator('[data-screen="complete"] button')).toHaveCount(0);

  // The researcher exports the CSV from the admin page.
  const csvPage = await context.newPage();
  await csvPage.goto(`${DEV}/admin.html`);
  await csvPage.locator("label.check input").check(); // development sessions are hidden by default
  await csvPage.locator(".sessions-table tbody tr").first().click();
  const csvDownload = csvPage.waitForEvent("download");
  await csvPage.getByRole("button", { name: "Download CSV" }).click();
  const rows = parseCsv(fs.readFileSync(await (await csvDownload).path(), "utf8"));
  await csvPage.close();
  expect(rows).toHaveLength(10);
  const drift = rows.filter((r) => r.phase === "drift");
  expect(drift.map((r) => r.currentAngleA)).toEqual(["0", "30", "60", "90", "120", "150"]);
  expect(drift.every((r) => r.feedbackShown === "false")).toBe(true);
  expect(rows[0]).toMatchObject({ developmentMode: "true", consentBypassed: "true", consentGiven: "false", sessionSeed: String(experimentConfig.randomization.fixedSeed), completionStatus: "complete" });
  const devOverrides = JSON.parse(rows[0].developmentOverridesJson);
  expect(devOverrides).toMatchObject({ "training.minTrials": 4, "drift.numberOfTrials": 6 });
  expect(devOverrides).not.toHaveProperty("sessionSeed");
  for (const r of rows) expect(Number(r.responseTimeMs)).toBeGreaterThan(0);

  // Plot export.
  const svgDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export SVG" }).click();
  const svgText = fs.readFileSync(await (await svgDownload).path(), "utf8");
  expect(svgText).toContain("<svg");
  await page.screenshot({ path: "test-results/dev-complete.png", fullPage: true });

  // Review page reads the saved session from localStorage.
  const review = await context.newPage();
  await review.goto(`${DEV}/review.html`);
  await expect(review.locator("select option").nth(1)).toContainText("complete");
  await review.locator("select").first().selectOption({ index: 1 });
  await expect(review.locator('[data-layer="dynamic"] circle[data-trial]')).toHaveCount(9); // most recent trial drawn as "current"
  await expect(review.locator(".status-line").nth(1)).toContainText("10 answered trials");
  await review.screenshot({ path: "test-results/review.png", fullPage: true });

  // Admin page (dev server: sessions from this browser's localStorage, no login).
  const admin = await context.newPage();
  await admin.goto(`${DEV}/admin.html`);
  await admin.locator("label.check input").check(); // development sessions are hidden by default
  await admin.locator(".sessions-table tbody tr").first().click();
  await expect(admin.locator(".image-grid .thumb")).toHaveCount(10);
  await expect(admin.locator(".image-grid-title").first()).toContainText("Training");
  await expect(admin.getByRole("button", { name: "Planned order for this seed" })).toHaveCount(0);
  await admin.getByRole("link", { name: "Planned image order" }).click();
  await expect(admin.locator(".seed-value")).toHaveText(String(experimentConfig.randomization.fixedSeed));
  await expect(admin.locator('input[aria-label="Seed"]')).toHaveCount(0); // read-only
  await expect(admin.locator(".image-grid .thumb")).toHaveCount(400); // default config: 40 training + 360 drift
  await admin.screenshot({ path: "test-results/admin.png", fullPage: true });
});
