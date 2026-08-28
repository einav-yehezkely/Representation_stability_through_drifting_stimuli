############################################################
# FILTER PCA DATA BY AVAILABLE ARCFACE EMBEDDINGS
#
# Takes:
#   pca_top2_filtered_female.csv
#
# Keeps only images that exist in:
#   female_arcface_embeddings.csv
#
# Creates:
#   pca_top2_filtered_female_arcface_1.csv
############################################################

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
ARCFACE_CSV = "female_arcface_embeddings.csv"

OUTPUT_CSV = "pca_top2_filtered_female_1.csv"


# ============================================================
# LOAD PCA FILE
# ============================================================

print("Loading PCA file...")

pca_df = pd.read_csv(
    PCA_CSV,
    header=None
)

print("Images in original PCA file:", len(pca_df))


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("Loading ArcFace embeddings...")

arcface_df = pd.read_csv(
    ARCFACE_CSV,
    header=None
)

print("Images with ArcFace embeddings:", len(arcface_df))


# ============================================================
# CREATE SET OF VALID ARCFACE FILENAMES
# ============================================================

arcface_names = set(
    arcface_df.iloc[:, 0]
    .astype(str)
    .apply(os.path.basename)
    .str.strip()
)


# ============================================================
# NORMALIZE PCA FILENAMES
# ============================================================

pca_names = (
    pca_df.iloc[:, 0]
    .astype(str)
    .apply(os.path.basename)
    .str.strip()
)


# ============================================================
# FIND WHICH PCA IMAGES HAVE AN ARCFACE EMBEDDING
# ============================================================

valid_mask = pca_names.isin(arcface_names)

filtered_df = pca_df[valid_mask].copy()


# ============================================================
# REPORT REMOVED IMAGES
# ============================================================

missing_names = pca_names[~valid_mask].tolist()

print()
print("=" * 60)
print("FILTERING RESULTS")
print("=" * 60)

print("PCA images before filtering:", len(pca_df))
print("PCA images after filtering: ", len(filtered_df))
print("Removed images:             ", len(missing_names))

if missing_names:
    print()
    print("Removed filenames:")

    for name in missing_names:
        print("  ", name)


# ============================================================
# SAVE NEW PCA FILE
#
# Keep exactly the same format as the original:
# filename, PC1, PC2
# No header and no index.
# ============================================================

filtered_df.to_csv(
    OUTPUT_CSV,
    header=False,
    index=False
)


print()
print("=" * 60)
print("Saved filtered PCA file:")
print(OUTPUT_CSV)
print("=" * 60)