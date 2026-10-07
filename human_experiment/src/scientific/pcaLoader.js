// PCA data loading. Counterpart of load_top2_filtered (classify_rotation_resnet50.py:236-251):
// rows are (filename, PC1, PC2) in source order; row index is the image index.

export class AssetError extends Error {
  constructor(message) {
    super(message);
    this.name = "AssetError";
  }
}

/** Build the in-memory PCA structure from the parsed pca.json payload. */
export function buildPca(payload, manifest = null) {
  const { imageIds, pc1, pc2 } = payload ?? {};
  if (!Array.isArray(imageIds) || !Array.isArray(pc1) || !Array.isArray(pc2)) {
    throw new AssetError("PCA data is malformed (expected imageIds, pc1, pc2 arrays).");
  }
  if (imageIds.length !== pc1.length || imageIds.length !== pc2.length || imageIds.length === 0) {
    throw new AssetError("PCA data arrays are empty or have different lengths.");
  }
  const x = Float64Array.from(pc1);
  const y = Float64Array.from(pc2);
  for (let i = 0; i < x.length; i += 1) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) throw new AssetError(`PCA row ${i + 1} has non-finite coordinates.`);
  }
  if (new Set(imageIds).size !== imageIds.length) throw new AssetError("PCA data contains duplicate image identifiers.");
  let maxRadius = 0;
  for (let i = 0; i < x.length; i += 1) maxRadius = Math.max(maxRadius, Math.sqrt(x[i] * x[i] + y[i] * y[i]));
  return {
    count: imageIds.length,
    imageIds,
    pc1: x,
    pc2: y,
    maxRadius,
    source: payload.source,
    sha256: payload.sha256,
    manifest,
  };
}

/** Fetch pca.json and the assets manifest. */
export async function loadPca(config, fetchImpl = fetch) {
  const get = async (path) => {
    let response;
    try {
      response = await fetchImpl(path);
    } catch (error) {
      throw new AssetError(`Could not load ${path}: ${error.message}`);
    }
    if (!response.ok) throw new AssetError(`Could not load ${path} (HTTP ${response.status}).`);
    return response.json();
  };
  const [payload, manifest] = await Promise.all([get(config.assets.pcaDataPath), get(config.assets.manifestPath)]);
  const pca = buildPca(payload, manifest);
  if (manifest.pcaSha256 !== pca.sha256) throw new AssetError("PCA data does not match the assets manifest.");
  return pca;
}
