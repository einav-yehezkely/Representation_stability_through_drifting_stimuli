"""
ShuffleNet embedding extraction, PCA, and filtering pipeline.

Extracts embeddings from all images in the image directory using a pretrained
ShuffleNetV2-x0.5, reduces dimensionality with PCA (95% variance), and filters
images by proximity to the PC1-PC2 plane in the residual PCA subspace.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ShuffleNet_V2_X0_5_Weights
from tqdm import tqdm

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── tuneable constants ─────────────────────────────────────────────────────────
FILTER_PERCENT: int = 10        # keep bottom N% by residual distance
BATCH_SIZE: int = 256           # DataLoader batch size
NUM_WORKERS: int = 0            # DataLoader workers (set 0 on Windows if issues arise)

# ── paths ──────────────────────────────────────────────────────────────────────
# All paths are relative to the project root (two levels up from this file).
SCRIPT_DIR = Path(__file__).resolve().parent          # …/shufflenetSpace
PROJECT_ROOT = SCRIPT_DIR.parent                       # project root

# Candidate image-directory names (checked in order)
_IMAGE_DIR_CANDIDATES = ["female_faces", "femle faces", "female faces", "faces"]

OUTPUT_DIR = SCRIPT_DIR                                # write outputs here

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    return device


def get_model(device: torch.device) -> Tuple[nn.Module, transforms.Compose]:
    """
    Load pretrained ShuffleNetV2-x0.5.

    Replaces model.fc with nn.Identity() so that forward() returns the
    penultimate feature vector instead of classification logits.

    Returns (model, preprocessing_transform).
    """
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = models.shufflenet_v2_x0_5(weights=weights)

    # Swap the final FC to obtain embeddings
    model.fc = nn.Identity()

    model.eval()
    model.to(device)

    preprocess = weights.transforms()  # identical to the official preprocessing
    log.info(
        "ShuffleNetV2-x0.5 loaded  |  embedding dim (before Identity): 1024"
    )
    return model, preprocess


def get_image_files(root: Path) -> Tuple[Path, List[Path]]:
    """
    Locate the image directory and return a sorted list of valid image paths.

    Tries _IMAGE_DIR_CANDIDATES in order; falls back to the first subdirectory
    that contains image files.
    """
    img_dir: Path | None = None
    for name in _IMAGE_DIR_CANDIDATES:
        candidate = root / name
        if candidate.is_dir():
            img_dir = candidate
            break

    if img_dir is None:
        # Generic fallback: first sub-directory with images
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                if any(f.suffix.lower() in VALID_EXTENSIONS for f in sub.iterdir()):
                    img_dir = sub
                    break

    if img_dir is None:
        raise FileNotFoundError(
            f"No suitable image directory found under {root}. "
            f"Tried: {_IMAGE_DIR_CANDIDATES}"
        )

    log.info("Image directory: %s", img_dir)

    files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
    )
    log.info("Found %d image files", len(files))
    return img_dir, files


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset + DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    """Loads images from a list of paths and applies a transform."""

    def __init__(self, paths: List[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path.name


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    model: nn.Module,
    image_paths: List[Path],
    transform: transforms.Compose,
    device: torch.device,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Run all images through the model in batches.

    Returns:
        filenames   – names of successfully processed images
        embeddings  – (N, D) float32 array
        failed      – list of filenames that raised exceptions
    """
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

    log.info("Extracting embeddings  (batch_size=%d, workers=%d) …", BATCH_SIZE, NUM_WORKERS)

    with torch.no_grad():
        for batch_imgs, batch_names in tqdm(loader, desc="Batches", unit="batch"):
            try:
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                feats = model(batch_imgs)          # (B, D)
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
        "Processed %d images  |  %d failed  |  embedding dim = %d",
        len(filenames), len(failed), embeddings.shape[1] if embeddings.ndim == 2 else 0,
    )
    return filenames, embeddings, failed


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(embeddings: np.ndarray) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA retaining 95% explained variance.

    Returns the fitted PCA object and the projected coordinates (N, k).
    """
    log.info("Fitting PCA (n_components=0.95) …")
    pca = PCA(n_components=0.95, random_state=SEED)
    coords = pca.fit_transform(embeddings)

    n_components = pca.n_components_
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    log.info("PCA summary:")
    log.info("  Images processed      : %d", embeddings.shape[0])
    log.info("  Original dimension    : %d", embeddings.shape[1])
    log.info("  Retained components   : %d", n_components)
    log.info("  Cumulative variance   : %.4f", cum_var[-1])
    log.info("  Explained var  PC1    : %.4f", pca.explained_variance_ratio_[0])
    if n_components >= 2:
        log.info("  Explained var  PC2    : %.4f", pca.explained_variance_ratio_[1])

    return pca, coords


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Residual distances and filtering
# ─────────────────────────────────────────────────────────────────────────────

def calculate_residual_distances(coords: np.ndarray) -> np.ndarray:
    """
    For each image compute sqrt(PC3² + PC4² + … + PC_k²).

    If only 2 components exist, returns zeros (all images kept).
    """
    if coords.shape[1] <= 2:
        log.info("Only 2 PCA components – residual distances set to 0 for all images.")
        return np.zeros(coords.shape[0], dtype=np.float32)

    residual = coords[:, 2:]                          # columns 3 … k
    distances = np.linalg.norm(residual, axis=1).astype(np.float32)
    return distances


def filter_images(
    filenames: List[str],
    coords: np.ndarray,
    residual_distances: np.ndarray,
    percent: int = FILTER_PERCENT,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Keep the bottom `percent`% of images by residual distance.

    Returns:
        sel_filenames       – selected filenames
        sel_coords          – (M, k) PCA coordinates of selected images
        sel_res_distances   – (M,)   residual distances of selected images
    """
    threshold = np.percentile(residual_distances, percent)
    mask = residual_distances <= threshold
    n_keep = int(mask.sum())

    log.info(
        "Filtering: keeping %.0f%% with residual_distance ≤ %.4f  → %d / %d images",
        percent, threshold, n_keep, len(filenames),
    )

    sel_filenames = [f for f, m in zip(filenames, mask) if m]
    sel_coords = coords[mask]
    sel_res = residual_distances[mask]
    return sel_filenames, sel_coords, sel_res


# ─────────────────────────────────────────────────────────────────────────────
# 6-7.  Saving results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    output_dir: Path,
    filenames: List[str],
    embeddings: np.ndarray,
    pca: PCA,
    coords: np.ndarray,
    residual_distances: np.ndarray,
    sel_filenames: List[str],
    sel_coords: np.ndarray,
    failed: List[str],
) -> None:
    """Write all output CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 6. Main filtered output ──────────────────────────────────────────────
    df_filtered = pd.DataFrame({
        "filename": sel_filenames,
        "PC1": sel_coords[:, 0],
        "PC2": sel_coords[:, 1],
    })
    path_filtered = output_dir / "shufflenet_pca_filtered.csv"
    df_filtered.to_csv(path_filtered, index=False)
    log.info("Saved: %s  (%d rows)", path_filtered.name, len(df_filtered))

    # ── 7a. Full embeddings ──────────────────────────────────────────────────
    n_dim = embeddings.shape[1]
    embed_cols = {f"emb_{i}": embeddings[:, i] for i in range(n_dim)}
    df_embed = pd.DataFrame({"filename": filenames, **embed_cols})
    path_embed = output_dir / "shufflenet_embeddings.csv"
    df_embed.to_csv(path_embed, index=False)
    log.info("Saved: %s  (%d rows, %d dims)", path_embed.name, len(df_embed), n_dim)

    # ── 7b. All PCA coordinates + residual distance ──────────────────────────
    n_comp = coords.shape[1]
    pca_cols = {f"PC{i+1}": coords[:, i] for i in range(n_comp)}
    df_pca = pd.DataFrame({
        "filename": filenames,
        **pca_cols,
        "residual_distance": residual_distances,
    })
    path_pca = output_dir / "shufflenet_pca_95.csv"
    df_pca.to_csv(path_pca, index=False)
    log.info("Saved: %s  (%d rows, %d PCs)", path_pca.name, len(df_pca), n_comp)

    # ── 7c. Variance table ───────────────────────────────────────────────────
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    df_var = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_comp)],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": cum_var,
    })
    path_var = output_dir / "shufflenet_pca_variance.csv"
    df_var.to_csv(path_var, index=False)
    log.info("Saved: %s  (%d components)", path_var.name, n_comp)

    # ── Failed images log ────────────────────────────────────────────────────
    if failed:
        path_failed = output_dir / "failed_images.txt"
        path_failed.write_text("\n".join(failed))
        log.warning("Logged %d failed images to %s", len(failed), path_failed.name)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def create_plots(
    output_dir: Path,
    filenames: List[str],
    coords: np.ndarray,
    sel_filenames: List[str],
    sel_coords: np.ndarray,
) -> None:
    """Save scatter plots for all images and for the filtered subset."""

    _ALPHA = 0.3
    _S = 4           # marker size
    _DPI = 150

    # ── All images ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=_S, alpha=_ALPHA, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"ShuffleNet  –  PC1 vs PC2  (all {len(filenames)} images)")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    path_all = output_dir / "shufflenet_pca_all.png"
    fig.savefig(path_all, dpi=_DPI)
    plt.close(fig)
    log.info("Saved: %s", path_all.name)

    # ── Filtered images ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(sel_coords[:, 0], sel_coords[:, 1], s=_S, alpha=_ALPHA,
               color="steelblue", linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"ShuffleNet  –  PC1 vs PC2  (filtered {FILTER_PERCENT}%: "
        f"{len(sel_filenames)} images)"
    )
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    path_filt = output_dir / "shufflenet_pca_filtered.png"
    fig.savefig(path_filt, dpi=_DPI)
    plt.close(fig)
    log.info("Saved: %s", path_filt.name)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("ShuffleNet embedding pipeline  –  FILTER_PERCENT=%d%%", FILTER_PERCENT)
    log.info("=" * 60)

    # 1. Device & model
    device = get_device()
    model, transform = get_model(device)

    # 2. Discover images
    _img_dir, image_paths = get_image_files(PROJECT_ROOT)

    # 3. Extract embeddings
    filenames, embeddings, failed = extract_embeddings(
        model, image_paths, transform, device
    )

    if len(filenames) == 0:
        log.error("No images were successfully processed. Exiting.")
        sys.exit(1)

    # 4. PCA
    pca, coords = run_pca(embeddings)

    # 5. Residual distances & filtering
    residual_distances = calculate_residual_distances(coords)
    sel_filenames, sel_coords, sel_res = filter_images(
        filenames, coords, residual_distances, percent=FILTER_PERCENT
    )

    # 6-7. Save CSVs
    save_results(
        output_dir=OUTPUT_DIR,
        filenames=filenames,
        embeddings=embeddings,
        pca=pca,
        coords=coords,
        residual_distances=residual_distances,
        sel_filenames=sel_filenames,
        sel_coords=sel_coords,
        failed=failed,
    )

    # 8. Plots
    create_plots(
        output_dir=OUTPUT_DIR,
        filenames=filenames,
        coords=coords,
        sel_filenames=sel_filenames,
        sel_coords=sel_coords,
    )

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  All outputs in: %s", OUTPUT_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
