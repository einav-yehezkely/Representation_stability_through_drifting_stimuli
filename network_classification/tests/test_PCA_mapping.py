############################################################
# TEST WHETHER FACENET PCA(2D) MAPS LINEARLY TO ARCFACE PCA(2D)
#
# Steps:
# 1. Load FaceNet PCA coordinates from existing PCA CSV
# 2. Load ArcFace 512D embeddings
# 3. Compute ArcFace PCA -> 2D
# 4. Match the same images
# 5. Learn a linear mapping:
#
#       FaceNet (PC1, PC2)
#               ↓
#         linear mapping
#               ↓
#       ArcFace (PC1, PC2)
#
# 6. Evaluate on unseen test images
############################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

ROOT_DIR = os.path.dirname(
    PROJECT_DIR
)


FACENET_PCA_CSV = os.path.join(
    ROOT_DIR,
    "pca_top2_filtered_female_1.csv"
)

ARCFACE_EMBEDDINGS_CSV = os.path.join(
    ROOT_DIR,
    "female_arcface_embeddings.csv"
)

OUTPUT_RESULTS_CSV = os.path.join(
    ROOT_DIR,
    "facenet_pca2_to_arcface_pca2_results.csv"
)

OUTPUT_MAPPING_GRAPH = os.path.join(
    ROOT_DIR,
    "facenet_pca2_to_arcface_pca2_mapping.png"
)

OUTPUT_ANGLE_GRAPH = os.path.join(
    ROOT_DIR,
    "arcface_pca_colored_by_facenet_angle.png"
)


# ============================================================
# LOAD FACENET PCA
# ============================================================

print("Loading FaceNet PCA...")

facenet_df = pd.read_csv(
    FACENET_PCA_CSV,
    header=None
)

facenet_df.columns = [
    "filename",
    "facenet_PC1",
    "facenet_PC2"
]

facenet_df["filename"] = (
    facenet_df["filename"]
    .astype(str)
    .apply(os.path.basename)
)

facenet_df["facenet_angle"] = (
    np.degrees(
        np.arctan2(
            facenet_df["facenet_PC2"],
            facenet_df["facenet_PC1"]
        )
    ) % 360
)

print(
    "FaceNet PCA images:",
    len(facenet_df)
)


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("Loading ArcFace embeddings...")

arc_df = pd.read_csv(
    ARCFACE_EMBEDDINGS_CSV,
    header=None
)

arc_names = (
    arc_df.iloc[:, 0]
    .astype(str)
    .apply(os.path.basename)
)

arc_embeddings = arc_df.iloc[
    :,
    1:513
].to_numpy(
    dtype=np.float32
)


# ============================================================
# L2 NORMALIZE ARCFACE
# ============================================================

norms = np.linalg.norm(
    arc_embeddings,
    axis=1,
    keepdims=True
)

norms[
    norms == 0
] = 1

arc_embeddings = (
    arc_embeddings / norms
)


# ============================================================
# COMPUTE ARCFACE PCA -> 2D
# ============================================================

print("Computing ArcFace PCA...")

arcface_pca = PCA(
    n_components=2
)

arcface_2d = arcface_pca.fit_transform(
    arc_embeddings
)


print()
print(
    "ArcFace PCA explained variance ratio:",
    arcface_pca.explained_variance_ratio_
)

print(
    "Total variance explained by ArcFace PC1+PC2:",
    arcface_pca.explained_variance_ratio_.sum()
)


# ============================================================
# BUILD ARCFACE PCA DATAFRAME
# ============================================================

arcface_df = pd.DataFrame(
    {
        "filename":
            arc_names,

        "arcface_PC1":
            arcface_2d[
                :,
                0
            ],

        "arcface_PC2":
            arcface_2d[
                :,
                1
            ],
    }
)


# ============================================================
# MATCH SAME IMAGES
# ============================================================

merged = facenet_df.merge(
    arcface_df,
    on="filename",
    how="inner"
)


print()
print(
    "Images existing in both spaces:",
    len(merged)
)


# ============================================================
# BUILD MATRICES
# ============================================================

X = merged[
    [
        "facenet_PC1",
        "facenet_PC2"
    ]
].to_numpy(
    dtype=np.float32
)

Y = merged[
    [
        "arcface_PC1",
        "arcface_PC2"
    ]
].to_numpy(
    dtype=np.float32
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(merged)
)

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)


X_train = X[
    train_idx
]

X_test = X[
    test_idx
]

Y_train = Y[
    train_idx
]

Y_test = Y[
    test_idx
]


print()
print(
    "Train images:",
    len(train_idx)
)

print(
    "Test images:",
    len(test_idx)
)


# ============================================================
# LEARN LINEAR MAPPING
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    Y_train
)


Y_pred = model.predict(
    X_test
)


# ============================================================
# R²
# ============================================================

r2_global = r2_score(
    Y_test,
    Y_pred,
    multioutput="variance_weighted"
)

r2_each = r2_score(
    Y_test,
    Y_pred,
    multioutput="raw_values"
)


# ============================================================
# CORRELATION
# ============================================================

corr_pc1 = np.corrcoef(
    Y_test[:, 0],
    Y_pred[:, 0]
)[
    0,
    1
]

corr_pc2 = np.corrcoef(
    Y_test[:, 1],
    Y_pred[:, 1]
)[
    0,
    1
]


# ============================================================
# EUCLIDEAN ERROR
# ============================================================

errors = np.linalg.norm(
    Y_test - Y_pred,
    axis=1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("FACENET PCA(2D) -> ARCFACE PCA(2D)")
print("=" * 70)

print()

print(
    "Global test R²:",
    r2_global
)

print(
    "ArcFace PC1 R²:",
    r2_each[0]
)

print(
    "ArcFace PC2 R²:",
    r2_each[1]
)

print()

print(
    "Correlation predicted/true ArcFace PC1:",
    corr_pc1
)

print(
    "Correlation predicted/true ArcFace PC2:",
    corr_pc2
)

print()

print(
    "Mean Euclidean prediction error:",
    np.mean(errors)
)

print(
    "Median Euclidean prediction error:",
    np.median(errors)
)

print()

print(
    "Linear transformation matrix:"
)

print(
    model.coef_
)

print()

print(
    "Intercept:"
)

print(
    model.intercept_
)

print("=" * 70)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_df = merged.iloc[
    test_idx
].copy()

test_df[
    "pred_arcface_PC1"
] = Y_pred[
    :,
    0
]

test_df[
    "pred_arcface_PC2"
] = Y_pred[
    :,
    1
]

test_df[
    "prediction_error"
] = errors


test_df.to_csv(
    OUTPUT_RESULTS_CSV,
    index=False
)


# ============================================================
# GRAPH 1:
# TRUE ARCFACE PCA VS LINEARLY PREDICTED ARCFACE PCA
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    Y_test[:, 0],
    Y_test[:, 1],
    s=10,
    alpha=0.35,
    label="True ArcFace PCA"
)

plt.scatter(
    Y_pred[:, 0],
    Y_pred[:, 1],
    s=10,
    alpha=0.35,
    label="Predicted from FaceNet PCA"
)

plt.xlabel(
    "ArcFace PC1"
)

plt.ylabel(
    "ArcFace PC2"
)

plt.title(
    "True ArcFace PCA vs Linear Prediction from FaceNet PCA"
)

plt.axis(
    "equal"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_MAPPING_GRAPH,
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2:
# ARCFACE PCA COLORED BY FACENET ANGLE
# ============================================================

plt.figure(
    figsize=(10, 8)
)

scatter = plt.scatter(
    merged[
        "arcface_PC1"
    ],
    merged[
        "arcface_PC2"
    ],
    c=merged[
        "facenet_angle"
    ],
    s=8,
    alpha=0.6,
    cmap="hsv"
)

plt.xlabel(
    "ArcFace PC1"
)

plt.ylabel(
    "ArcFace PC2"
)

plt.title(
    "ArcFace PCA Colored by FaceNet PCA Angle"
)

plt.axis(
    "equal"
)

plt.grid(
    True,
    alpha=0.3
)

colorbar = plt.colorbar(
    scatter
)

colorbar.set_label(
    "Angle in FaceNet PCA space (degrees)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ANGLE_GRAPH,
    dpi=300
)

plt.close()


print()
print(
    "Saved results:",
    OUTPUT_RESULTS_CSV
)

print(
    "Saved mapping graph:",
    OUTPUT_MAPPING_GRAPH
)

print(
    "Saved angle graph:",
    OUTPUT_ANGLE_GRAPH
)