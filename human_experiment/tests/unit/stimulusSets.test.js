// Switching the image set on the admin page (assets.stimulusSet): configuration, validation,
// image size from the manifest, and the prepared assets of both sets.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { experimentConfig } from "../../src/config/experimentConfig.js";
import { applySettings, findSettingsProblems, settingsFromConfig, stimulusSetOf, STIMULUS_SET_KEY } from "../../src/config/experimentSettings.js";
import { findConfigProblems, withImageSize } from "../../src/config/validateConfig.js";
import { buildPca } from "../../src/scientific/pcaLoader.js";

const assetsDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "public");
const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(assetsDir, rel), "utf8"));
const settings = (patch = {}) => ({ ...settingsFromConfig(experimentConfig), ...patch });

describe("image sets", () => {
  it("the original set is the default, also for settings saved before the switch existed", () => {
    expect(settingsFromConfig(experimentConfig)[STIMULUS_SET_KEY]).toBe("original");
    const legacy = { "training.initialAngleA": 10, "training.initialAngleB": 190, "drift.degreesPerTrial": 1, "drift.numberOfTrials": 360 };
    const config = applySettings(experimentConfig, legacy);
    expect(config.assets.stimulusSet).toBe("original");
    expect(config.assets.pcaDataPath).toBe("assets/pca.json");
    expect(findSettingsProblems(legacy, experimentConfig, { pcaCount: 11817 })).toEqual([]);
  });

  it("choosing the morph set switches the faces folder, PCA file and manifest", () => {
    const config = applySettings(experimentConfig, settings({ [STIMULUS_SET_KEY]: "morph" }));
    expect(config.assets).toMatchObject({ stimulusSet: "morph", facesDirectory: "assets/faces_morph/", pcaDataPath: "assets/pca_morph.json", manifestPath: "assets/assets-manifest_morph.json" });
    expect(stimulusSetOf(experimentConfig, settings({ [STIMULUS_SET_KEY]: "morph" })).key).toBe("morph");
    expect(findConfigProblems(config)).toEqual([]);
  });

  it("rejects an unknown set and more drift trials than the set has images", () => {
    expect(findSettingsProblems(settings({ [STIMULUS_SET_KEY]: "nope" }), experimentConfig).join(" ")).toMatch(/Unknown image set/);
    const morph = readJson("assets/assets-manifest_morph.json");
    expect(findSettingsProblems(settings({ [STIMULUS_SET_KEY]: "morph", "drift.numberOfTrials": morph.pcaRowCount + 1 }), experimentConfig, { pcaCount: morph.pcaRowCount }).join(" ")).toMatch(/exceeds/);
  });

  it("takes the image size from the manifest and keeps the on-screen width", () => {
    const morph = withImageSize(applySettings(experimentConfig, settings({ [STIMULUS_SET_KEY]: "morph" })), { imageWidth: 512, imageHeight: 512 });
    expect(morph.assets.imageWidth).toBe(512);
    expect(morph.ui.imageDisplayScale * 512).toBeCloseTo(experimentConfig.ui.imageDisplayWidth);
    const original = withImageSize(applySettings(experimentConfig, null), { imageWidth: 178, imageHeight: 218 });
    expect(original.ui.imageDisplayScale).toBe(2); // unchanged for the original faces
    expect(withImageSize(original, {})).toBe(original); // older manifests without a size
  });

  it("both sets are prepared: PCA data matches its manifest, which records the image size", () => {
    for (const [key, set] of Object.entries(experimentConfig.stimulusSets)) {
      const manifest = readJson(set.manifestPath);
      const pca = buildPca(readJson(set.pcaDataPath), manifest);
      expect(manifest.stimulusSet, key).toBe(key);
      expect(manifest.pcaSha256).toBe(pca.sha256);
      expect(manifest.pcaRowCount).toBe(pca.count);
      expect(Number.isInteger(manifest.imageWidth) && Number.isInteger(manifest.imageHeight)).toBe(true);
    }
  });
});
