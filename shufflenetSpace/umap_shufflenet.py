"""
ShuffleNet → UMAP pipeline.

Loads (or extracts) ShuffleNetV2-x0.5 embeddings, optionally pre-reduces with
PCA, then projects to 2-D with UMAP and saves a scatter plot + CSV.

Smart caching: if shufflenet_embeddings.csv already exists in the output
directory (produced by extract_shufflenet_space.py), embeddings are loaded
from disk and the GPU step is skipped entirely.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ShuffleNet_V2_X0_5_Weights
from tqdm import tqdm

# ── reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── tuneable constants ─────────────────────────────────────────────────────────
# PCA pre-reduction before UMAP (speeds up UMAP on high-dim embeddings).
# Set to None to skip PCA and feed raw embeddings directly to UMAP.
PCA_PREREDUCE_DIM: int | None = 50

# UMAP hyperparameters
UMAP_N_NEIGHBORS: int = 15      # local neighbourhood size (larger → more global)
UMAP_MIN_DIST: float = 0.1      # minimum distance between points in 2-D layout
UMAP_METRIC: str = "cosine"     # distance metric in embedding space

# DataLoader settings (only used when re-extracting embeddings)
BATCH_SIZE: int = 256
NUM_WORKERS: int = 0            # keep 0 on Windows

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # …/shufflenetSpace
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = SCRIPT_DIR

_IMAGE_DIR_CANDIDATES = ["female_faces", "femle faces", "female faces", "faces"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "umap_pipeline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with extract_shufflenet_space.py
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    return device


def get_model(device: torch.device) -> Tuple[nn.Module, transforms.Compose]:
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = models.shufflenet_v2_x0_5(weights=weights)
    model.fc = nn.Identity()    # return pre-FC feature vector
    model.eval().to(device)
    preprocess = weights.transforms()
    log.info("ShuffleNetV2-x0.5 loaded (embedding dim 1024)")
    return model, preprocess


def get_image_files(root: Path) -> Tuple[Path, List[Path]]:
    img_dir: Path | None = None
    for name in _IMAGE_DIR_CANDIDATES:
        candidate = root / name
        if candidate.is_dir():
            img_dir = candidate
            break
    if img_dir is None:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and any(
                f.suffix.lower() in VALID_EXTENSIONS for f in sub.iterdir()
            ):
                img_dir = sub
                break
    if img_dir is None:
        raise FileNotFoundError(f"No image directory found under {root}")
    log.info("Image directory: %s", img_dir)
    files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS)
    log.info("Found %d image files", len(files))
    return img_dir, files


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.name


def extract_embeddings(
    model: nn.Module,
    image_paths: List[Path],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[List[str], np.ndarray, List[str]]:
    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=(NUM_WORKERS > 0),
    )

    filenames: List[str] = []
    embeddings_list: List[np.ndarray] = []
    failed: List[str] = []

    log.info("Extracting embeddings (batch_size=%d) …", BATCH_SIZE)
    with torch.no_grad():
        for batch_imgs, batch_names in tqdm(loader, desc="Batches", unit="batch"):
            try:
                feats = model(batch_imgs.to(device, non_blocking=True))
                feats_np = feats.cpu().numpy().astype(np.float32)
                for name, vec in zip(batch_names, feats_np):
                    filenames.append(name)
                    embeddings_list.append(vec)
            except Exception as exc:  # noqa: BLE001
                for name in batch_names:
                    warnings.warn(f"[SKIP] {name}: {exc}")
                    failed.append(name)

    embeddings = np.vstack(embeddings_list) if embeddings_list else np.empty((0, 0))
    log.info(
        "Extracted %d embeddings  |  %d failed  |  dim=%d",
        len(filenames), len(failed), embeddings.shape[1] if embeddings.ndim == 2 else 0,
    )
    return filenames, embeddings, failed


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings: load from cache or extract fresh
# ─────────────────────────────────────────────────────────────────────────────

def load_or_extract_embeddings(
    output_dir: Path,
    project_root: Path,
) -> Tuple[List[str], np.ndarray]:
    """
    Return (filenames, embeddings).

    If shufflenet_embeddings.csv exists in output_dir, load from there.
    Otherwise run the full extraction pipeline and save the CSV.
    """
    cache_path = output_dir / "shufflenet_embeddings.csv"

    if cache_path.exists():
        log.info("Loading cached embeddings from %s …", cache_path.name)
        df = pd.read_csv(cache_path)
        filenames = df["filename"].tolist()
        embeddings = df.drop(columns=["filename"]).values.astype(np.float32)
        log.info("Loaded %d embeddings  |  dim=%d", len(filenames), embeddings.shape[1])
        return filenames, embeddings

    log.info("No cached embeddings found – extracting from images …")
    device = get_device()
    model, transform = get_model(device)
    _img_dir, image_paths = get_image_files(project_root)
    filenames, embeddings, failed = extract_embeddings(model, image_paths, transform, device)

    if len(filenames) == 0:
        log.error("No images processed. Exiting.")
        sys.exit(1)

    # Save for future runs
    n_dim = embeddings.shape[1]
    df = pd.DataFrame({"filename": filenames, **{f"emb_{i}": embeddings[:, i] for i in range(n_dim)}})
    df.to_csv(cache_path, index=False)
    log.info("Saved embeddings cache: %s", cache_path.name)

    if failed:
        (output_dir / "failed_images.txt").write_text("\n".join(failed))
        log.warning("%d images failed (see failed_images.txt)", len(failed))

    return filenames, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Optional PCA pre-reduction
# ─────────────────────────────────────────────────────────────────────────────

def pca_prereduce(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """Reduce embeddings to n_components with PCA before passing to UMAP."""
    actual = min(n_components, embeddings.shape[1], embeddings.shape[0])
    log.info("PCA pre-reduction: %d → %d dims …", embeddings.shape[1], actual)
    pca = PCA(n_components=actual, random_state=SEED)
    reduced = pca.fit_transform(embeddings)
    cum_var = float(np.sum(pca.explained_variance_ratio_))
    log.info("PCA pre-reduction done  |  cumulative variance retained: %.4f", cum_var)
    return reduced.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Project embeddings to 2-D with UMAP.

    Returns (N, 2) array [UMAP1, UMAP2].
    """
    log.info(
        "Running UMAP  |  n_neighbors=%d  min_dist=%.2f  metric=%s  input_shape=%s …",
        UMAP_N_NEIGHBORS, UMAP_MIN_DIST, UMAP_METRIC, embeddings.shape,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(embeddings)
    log.info("UMAP done  |  output shape: %s", coords_2d.shape)
    return coords_2d.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    output_dir: Path,
    filenames: List[str],
    coords_2d: np.ndarray,
) -> None:
    df = pd.DataFrame({
        "filename": filenames,
        "UMAP1": coords_2d[:, 0],
        "UMAP2": coords_2d[:, 1],
    })
    path = output_dir / "shufflenet_umap.csv"
    df.to_csv(path, index=False)
    log.info("Saved: %s  (%d rows)", path.name, len(df))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def create_plot(
    output_dir: Path,
    filenames: List[str],
    coords_2d: np.ndarray,
) -> None:
    """Scatter plot of the full UMAP layout."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        s=3, alpha=0.25, linewidths=0,
        c=coords_2d[:, 0],          # colour by UMAP1 position for visual clarity
        cmap="viridis",
    )
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(
        f"ShuffleNetV2-x0.5  –  UMAP 2-D projection\n"
        f"{len(filenames):,} images  |  "
        f"n_neighbors={UMAP_N_NEIGHBORS}  min_dist={UMAP_MIN_DIST}  metric={UMAP_METRIC}",
        fontsize=11,
    )
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    path = output_dir / "shufflenet_umap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved: %s", path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("ShuffleNet → UMAP pipeline")
    log.info(
        "  PCA pre-reduce : %s",
        f"{PCA_PREREDUCE_DIM} dims" if PCA_PREREDUCE_DIM else "disabled",
    )
    log.info("  UMAP n_neighbors: %d", UMAP_N_NEIGHBORS)
    log.info("  UMAP min_dist   : %.2f", UMAP_MIN_DIST)
    log.info("  UMAP metric     : %s", UMAP_METRIC)
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Embeddings (cached or fresh)
    filenames, embeddings = load_or_extract_embeddings(OUTPUT_DIR, PROJECT_ROOT)

    # 2. Optional PCA pre-reduction
    if PCA_PREREDUCE_DIM is not None:
        embeddings = pca_prereduce(embeddings, PCA_PREREDUCE_DIM)

    # 3. UMAP projection
    coords_2d = run_umap(embeddings)

    # 4. Save CSV
    save_results(OUTPUT_DIR, filenames, coords_2d)

    # 5. Plot
    create_plot(OUTPUT_DIR, filenames, coords_2d)

    log.info("=" * 60)
    log.info("Done. Outputs in: %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
