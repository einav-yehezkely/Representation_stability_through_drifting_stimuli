// Single entry point for every experimental parameter.
//
// Scientific values marked "source" are derived from
// network_classification/classify_rotation_resnet50.py (copied verbatim to reference_source/).
// Values marked "researcher" are the design decisions listed in README.md.

export const experimentConfig = {
  experimentVersion: "2.0.0",
  schemaVersion: "2",

  drift: {
    initialAngleA: 0, // degrees
    initialAngleB: 180, // must be initialAngleA + 180 (B = -A)
    numberOfTrials: 360,
    degreesPerTrial: 1,
  },

  timing: {
    sampleDurationMs: 1000, // researcher: the face to compare with is shown for 1 s
    blankDurationMs: 500, // researcher: empty screen between that face and the two choices
    interTrialDurationMs: 500, // researcher: empty screen between the click and the next sample
  },

  assets: {
    facesDirectory: "assets/faces/", // served copy of female_faces/ (scripts/prepare-assets.mjs)
    pcaDataPath: "assets/pca.json", // derived from pca_top2_filtered_female_vgg_1.csv (scripts/prepare-assets.mjs)
    manifestPath: "assets/assets-manifest.json",
  },

  scientific: {
    origin: [0, 0], // source: angles are atan2 around the origin (lines 52-56)
    angleConvention: "atan2(PC2,PC1) in degrees, mod 360", // source: ANGLE_MAP (lines 52-56)
    rotationDirection: "counterclockwise", // source: rotate_vector (lines 300-318)
    targetRadius: 0.45, // source: create_base_and_opposite_points (line 276)
    radiusErrorWeight: 100, // source: combined_error = angle_error + radius_error * 100 (line 288)
    clusterSize: 1, // one image per selection (source NUM_OF_IMAGES_PER_CLUSTER = 64)
    clusterDistance: "euclidean2D", // source: collect_nearest_images (line 345)
    centerB: "negateA", // source: opposite_point = -base_point (line 295)
    selection: "nearestUnused", // researcher: no image is selected twice in a session
  },

  randomization: {
    // Every session uses this seed, so all participants see the same images in the same order.
    // It can only be changed here (not in development mode or on the admin page).
    seedPolicy: "fixed",
    fixedSeed: 20261014, // unsigned 32-bit integer
    algorithm: "sfc32",
    algorithmVersion: "1",
  },

  ui: {
    imageDisplayScale: 2, // source images are 178x218 px
    preloadCount: 3, // images of the next trials downloaded in advance (performance only; 0 disables)
  },

  storage: {
    // "server": every record is sent to save.php (online collection). The Vite dev server
    // (`npm run dev`) cannot run PHP, so it uses developmentAdapter instead.
    adapter: "server",
    developmentAdapter: "localCsv",
    serverEndpoint: "save.php", // relative to index.html
    requestTimeoutMs: 15000,
    maxAttempts: 3, // per save, before the participant sees "Try again"
    localStoragePrefix: "hx2", // differs from the first experiment, which uses "hx"
  },

  completion: {
    // Prolific completion link shown on the final screen, e.g.
    // "https://app.prolific.com/submissions/complete?cc=XXXXXXX". Empty: no link is shown.
    prolificCompletionUrl: "",
  },

  development: {
    // Off by default. When false, development mode can still be opened while running the Vite
    // dev server (`npm run dev`) with ?dev=1; it can never be opened in a production build
    // unless this flag is set to true.
    enabled: false,
  },
};
