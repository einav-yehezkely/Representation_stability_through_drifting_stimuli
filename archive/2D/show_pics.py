from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PCA_FILE = BASE_DIR / "pca_top2_female_generated_vgg.csv"
IMAGES_DIR = BASE_DIR / "female_generated"

THUMBNAIL_SIZE = 60
SEED = 42

# ============================================================
# LOAD PCA
# CSV without header: filename, PC1, PC2
# ============================================================

df = pd.read_csv(
    PCA_FILE,
    header=None,
    names=["filename", "PC1", "PC2"],
)

N_SAMPLE_IMAGES = len(df)

if df.empty:
    raise ValueError("The PCA file is empty.")

coordinates = df[["PC1", "PC2"]].to_numpy(dtype=float)

if not np.isfinite(coordinates).all():
    raise ValueError("The PCA file contains NaN or infinite coordinates.")

print(f"Loaded {len(df)} images.")

# Shared limits for both plots
lower = coordinates.min(axis=0)
upper = coordinates.max(axis=0)
padding = np.maximum((upper - lower) * 0.15, 0.05)


def configure_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)


# ============================================================
# GRAPH 1: ALL POINTS
# ============================================================

fig1, ax1 = plt.subplots(figsize=(10, 10))

ax1.scatter(
    df["PC1"],
    df["PC2"],
    s=20,
    alpha=0.6,
    color="gray",
)

configure_axes(ax1, f"Full PCA space (n={len(df)})")
fig1.tight_layout()

# ============================================================
# SELECT IMAGES
# Sample only from files that exist
# ============================================================

available_images = []

for index, filename in enumerate(df["filename"]):
    image_path = IMAGES_DIR / str(filename)

    if image_path.is_file():
        available_images.append((index, image_path))
    else:
        print(f"Missing image: {image_path}")

rng = np.random.default_rng(SEED)

sample_indices = rng.choice(
    len(available_images),
    size=min(N_SAMPLE_IMAGES, len(available_images)),
    replace=False,
)

# ============================================================
# GRAPH 2: ALL POINTS + SAMPLE IMAGES
# ============================================================

fig2, ax2 = plt.subplots(figsize=(14, 14))

ax2.scatter(
    df["PC1"],
    df["PC2"],
    s=15,
    alpha=0.3,
    color="gray",
)

displayed = 0

for sample_index in sample_indices:
    row_index, image_path = available_images[sample_index]
    row = df.iloc[row_index]

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail(
                (THUMBNAIL_SIZE, THUMBNAIL_SIZE),
                Image.Resampling.LANCZOS,
            )
            image_array = np.array(img)

    except OSError as error:
        print(f"Could not load {image_path.name}: {error}")
        continue

    image_box = OffsetImage(
        image_array,
        zoom=1,
        dpi_cor=False,
    )

    annotation = AnnotationBbox(
        image_box,
        (row["PC1"], row["PC2"]),
        frameon=True,
        pad=0.05,
        bboxprops={
            "edgecolor": "gray",
            "linewidth": 0.5,
            "facecolor": "white",
        },
    )

    ax2.add_artist(annotation)
    displayed += 1

configure_axes(
    ax2,
    f"Full PCA space with {displayed} sample images",
)

fig2.tight_layout()

fig1.savefig(
    BASE_DIR / "pca_all_points.png",
    dpi=300,
    bbox_inches="tight",
)

fig2.savefig(
    BASE_DIR / "pca_with_images.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()