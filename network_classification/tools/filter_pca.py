############################################################
# FILTER PCA DATA BY AVAILABLE RESNET50 EMBEDDINGS
#
# Takes:
#   pca_top2_filtered_female_vgg.csv
#
# Keeps only images that exist in:
#   female_resnet50_vggface2_embeddings.csv
#
# Creates:
#   pca_top2_filtered_female_vgg_1.csv
############################################################

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PCA_CSV = "pca_top2_filtered_female_vgg.csv"
RESNET50_CSV = "female_resnet50_vggface2_embeddings.csv"

OUTPUT_CSV = "pca_top2_filtered_female_vgg_1.csv"


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
# LOAD RESNET50 EMBEDDINGS
# ============================================================

print("Loading ResNet50 embeddings...")

resnet50_df = pd.read_csv(
    RESNET50_CSV,
    header=None
)

print("Images with ResNet50 embeddings:", len(resnet50_df))


# ============================================================
# CREATE SET OF VALID RESNET50 FILENAMES
# ============================================================

resnet50_names = set(
    resnet50_df.iloc[:, 0]
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
# FIND WHICH PCA IMAGES HAVE AN RESNET50 EMBEDDING
# ============================================================

valid_mask = pca_names.isin(resnet50_names)

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