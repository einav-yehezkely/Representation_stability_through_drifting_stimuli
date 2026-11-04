// Real-browser smoke tests (Chrome). Run with `npm run build && npm run test:e2e`.
import fs from "node:fs";
import { expect, test } from "@playwright/test";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { parseCsv } from "../../src/services/dataStorage/csvSerializer.js";

const PROD = "http://localhost:4174";
const DEV = "http://localhost:5174";

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
    const index = JSON.parse(localStorage.getItem("hx2:index") ?? "[]");
    const id = index.at(-1);
    return {
      meta: JSON.parse(localStorage.getItem(`hx2:session:${id}:meta`)),
      trials: JSON.parse(localStorage.getItem(`hx2:session:${id}:trials`) ?? "[]"),
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

const faces = (page) => page.locator("button.choice");

test("participant mode (production build): no dev controls, sample then two faces, server saving, reload", async ({ page }) => {
  const requests = await fakeSaveEndpoint(page);
  // ?dev=1 must be ignored by a production build.
  await page.goto(`${PROD}/index.html?PROLIFIC_PID=E2E-PID&dev=1`);
  await expect(page.locator("[data-dev]")).toHaveCount(0);
  await giveConsent(page);

  let previousSources = null;
  for (let i = 0; i < 4; i += 1) {
    // The sample alone.
    const sample = page.locator('[data-role="sample"] img');
    await expect(sample).toHaveCount(1);
    await expect(page.locator("img")).toHaveCount(1);
    await expect(page.locator(".pair")).toBeHidden();
    expect(await sample.evaluate((img) => img.complete && img.naturalWidth)).toBe(178);
    const sampleSource = await sample.getAttribute("src");
    if (previousSources) expect(previousSources).toContain(sampleSource); // continuity

    // Two faces, side by side, clickable.
    await expect(faces(page).first()).toBeEnabled();
    await expect(page.locator("img")).toHaveCount(2);
    await expect(page.locator('[data-role="sample"]')).toBeHidden();
    await expect(page.locator(".trial-prompt")).toBeVisible();
    const [left, right] = await Promise.all([faces(page).nth(0).boundingBox(), faces(page).nth(1).boundingBox()]);
    expect(left.x + left.width).toBeLessThan(right.x);
    expect(Math.abs(left.y - right.y)).toBeLessThan(1);
    previousSources = await page.locator("button.choice img").evaluateAll((imgs) => imgs.map((img) => img.getAttribute("src")));
    await faces(page).nth(i % 2).click();
    await expect(page.locator("img")).toHaveCount(0); // empty screen before the next sample
  }
  await expect(page.locator(".dev-panel")).toHaveCount(0);
  await expect(page.getByText(/^(Correct|Incorrect)$/)).toHaveCount(0);

  // Every answer was sent to the server.
  await expect.poll(() => requests.length).toBeGreaterThanOrEqual(5);
  const [startRequest, ...trialRequests] = requests;
  expect(startRequest.action).toBe("start");
  expect(startRequest.session).toMatchObject({ participantId: "E2E-PID", participantIdSource: "prolific_url", consentGiven: true, developmentMode: false, sessionSeed: experimentConfig.randomization.fixedSeed });
  expect(trialRequests).toHaveLength(4);
  for (const { action, trial } of trialRequests) {
    expect(action).toBe("trial");
    expect(trial.sessionId).toBe(startRequest.session.sessionId);
    expect(trial.responseTimeMs).toBeGreaterThan(0);
    expect(trial.responseTimeMs).toBeLessThan(5000);
    // Measured in the browser: 1000 ms and 500 ms (twice), plus up to a few frames.
    expect(trial.sampleDurationMs).toBeGreaterThanOrEqual(995);
    expect(trial.sampleDurationMs).toBeLessThan(1120);
    expect(trial.blankDurationMs).toBeGreaterThanOrEqual(495);
    expect(trial.blankDurationMs).toBeLessThan(620);
    if (trial.trialNumber > 1) {
      expect(trial.interTrialDurationMs).toBeGreaterThanOrEqual(495);
      expect(trial.interTrialDurationMs).toBeLessThan(620);
    }
  }
  expect(trialRequests[0].trial).toMatchObject({ sampleGroup: "A", interTrialDurationMs: null });
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

test("development mode: overrides, bypass, full run to CSV, live PCA plot, review and admin pages", async ({ page, context }) => {
  await page.goto(`${DEV}/index.html?dev=1`);
  await expect(page.locator('[data-screen="dev-setup"]')).toBeVisible();
  await page.locator('[data-field="drift.numberOfTrials"]').fill("6");
  await page.locator('[data-field="drift.degreesPerTrial"]').fill("30");
  await page.locator('[data-field="timing.sampleDurationMs"]').fill("100");
  await page.locator('[data-field="timing.blankDurationMs"]').fill("50");
  await expect(page.locator('[data-field="seed"]')).toHaveCount(0); // the seed is fixed, not editable
  await page.getByRole("button", { name: "Start session" }).click();
  await page.locator('[data-dev="bypass-consent"]').click();
  await page.getByRole("button", { name: "Start" }).click();

  await expect(faces(page).first()).toBeEnabled();
  await page.screenshot({ path: "test-results/dev-trial.png", fullPage: true });

  // PCA plot in the dev panel.
  await page.getByRole("button", { name: "PCA plot" }).click();
  await expect(page.locator('.dev-plot [data-layer="all-points"] circle')).toHaveCount(11817);

  const downloads = [];
  page.on("download", (d) => downloads.push(d.suggestedFilename()));
  for (let i = 0; i < 6; i += 1) {
    await expect(faces(page).first()).toBeEnabled();
    await faces(page).nth(i % 2).click();
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
  expect(rows).toHaveLength(6);
  expect(rows.map((r) => r.currentAngleA)).toEqual(["0", "30", "60", "90", "120", "150"]);
  expect(rows.map((r) => r.response)).toEqual(["left", "right", "left", "right", "left", "right"]);
  for (let i = 1; i < rows.length; i += 1) expect([rows[i - 1].optionAImageId, rows[i - 1].optionBImageId]).toContain(rows[i].sampleImageId);
  expect(rows[0]).toMatchObject({ sampleGroup: "A", sampleSourceTrial: "", developmentMode: "true", consentBypassed: "true", consentGiven: "false", sessionSeed: String(experimentConfig.randomization.fixedSeed), completionStatus: "complete" });
  const devOverrides = JSON.parse(rows[0].developmentOverridesJson);
  expect(devOverrides).toMatchObject({ "drift.numberOfTrials": 6, "timing.sampleDurationMs": 100 });
  expect(devOverrides).not.toHaveProperty("sessionSeed");
  for (const r of rows) expect(Number(r.responseTimeMs)).toBeGreaterThan(0);

  // Plot export.
  const svgDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export SVG" }).click();
  const svgText = fs.readFileSync(await (await svgDownload).path(), "utf8");
  expect(svgText).toContain("<svg");

  // Review page reads the saved session from localStorage.
  const review = await context.newPage();
  await review.goto(`${DEV}/review.html`);
  await expect(review.locator("select option").nth(1)).toContainText("complete");
  await review.locator("select").first().selectOption({ index: 1 });
  await expect(review.locator('[data-layer="dynamic"] circle[data-trial]')).toHaveCount(5); // most recent trial drawn as "current"
  await expect(review.locator(".status-line").nth(1)).toContainText("6 answered trials");

  // Admin page (dev server: sessions from this browser's localStorage, no login).
  const admin = await context.newPage();
  await admin.goto(`${DEV}/admin.html`);
  await admin.locator("label.check input").check(); // development sessions are hidden by default
  await admin.locator(".sessions-table tbody tr").first().click();
  await expect(admin.locator(".image-grid .trial-thumb")).toHaveCount(6);
  await expect(admin.locator(".trial-face.is-chosen")).toHaveCount(6);
  await admin.locator(".image-grid .trial-thumb").first().click();
  await expect(admin.locator(".trial-detail-face img")).toHaveCount(3);
  await admin.screenshot({ path: "test-results/admin-session.png", fullPage: true });
  await admin.getByRole("link", { name: "Planned image order" }).click();
  await expect(admin.locator(".seed-value")).toHaveText(String(experimentConfig.randomization.fixedSeed));
  await expect(admin.locator('input[aria-label="Seed"]')).toHaveCount(0); // read-only
  await expect(admin.locator(".image-grid .trial-thumb")).toHaveCount(360);
  await admin.getByRole("link", { name: "Settings" }).click();
  await expect(admin.locator('[data-setting="drift.initialAngleA"]')).toHaveValue("0");
  await admin.getByRole("button", { name: "Compare the two groups" }).click();
  await expect(admin.locator(".compare-column .thumb").first()).toBeVisible();
});
