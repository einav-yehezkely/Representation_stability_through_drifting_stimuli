// Single entry point for every experimental parameter (PRD §10).
//
// Scientific values marked "source" are derived from
// network_classification/classify_rotation_resnet50.py (copied verbatim to
// human_experiment/reference_source/). Values marked "decision Dn" are researcher decisions
// recorded in IMPLEMENTATION_PLAN.md §2.1. Values marked "placeholder OQn" are not specified
// by the PRD or the source and are documented as open questions (IMPLEMENTATION_PLAN.md §2.3).

export const experimentConfig = {
  experimentVersion: "1.0.0",
  schemaVersion: "1",

  training: {
    initialAngleA: 0, // degrees; also the initial drift centers (no duplicate drift setting)
    initialAngleB: 180, // must be initialAngleA + 180 (decision D3)
    minTrials: 20,
    accuracyWindow: 20,
    accuracyThreshold: 0.8,
  },

  drift: {
    numberOfTrials: 360,
    degreesPerTrial: 1,
  },

  assets: {
    facesDirectory: "assets/faces/", // served copy of female_faces/ (scripts/prepare-assets.mjs)
    pcaDataPath: "assets/pca.json", // derived from pca_top2_filtered_female_vgg_1.csv
    manifestPath: "assets/assets-manifest.json",
  },

  scientific: {
    origin: [0, 0], // source: angles are atan2 around the origin (lines 52-56)
    angleConvention: "atan2(PC2,PC1) in degrees, mod 360", // source: ANGLE_MAP (lines 52-56)
    rotationDirection: "counterclockwise", // source: rotate_vector (lines 300-318)
    targetRadius: 0.45, // source: create_base_and_opposite_points (line 276)
    radiusErrorWeight: 100, // source: combined_error = angle_error + radius_error * 100 (line 288)
    clusterSize: 1, // decision D1 (source NUM_OF_IMAGES_PER_CLUSTER = 64)
    clusterDistance: "euclidean2D", // source: collect_nearest_images (line 345)
    centerB: "negateA", // decision D3; source: opposite_point = -base_point (line 295)
    trainingSelection: "nearestUnused", // decision D4; source used_indices pattern (lines 795-802)
    driftSelection: "nearestUnused", // decision D5
    exclusionScope: "perPhase", // decisions D4 + D5: training and drift keep separate used sets
  },

  randomization: {
    seedPolicy: "crypto-random-uint32",
    algorithm: "sfc32",
    algorithmVersion: "1",
    trainingBlockSize: 4, // placeholder OQ3: balanced shuffled blocks of 2 A + 2 B
  },

  ui: {
    feedbackDurationMs: 1000, // placeholder OQ3
    imageDisplayScale: 2, // placeholder OQ3: source images are 178x218 px
    feedbackText: { correct: "Correct", incorrect: "Incorrect" },
  },

  storage: {
    adapter: "localCsv",
    localStoragePrefix: "hx",
  },

  development: {
    // Off by default. When false, development mode can still be opened while running the Vite
    // dev server (`npm run dev`) with ?dev=1; it can never be opened in a production build
    // unless this flag is set to true.
    enabled: false,
  },
};
