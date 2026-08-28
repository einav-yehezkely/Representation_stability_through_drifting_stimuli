############################################################
# TEST WHETHER THE TOP-2 FACENET PCA SUBSPACE
# IS PRESERVED IN ARCFACE
#
# We test:
#
# (PC1, PC2) of FaceNet
#        ↓
# linear mapping
#        ↓
# ArcFace 512D embedding
#
# Evaluation is performed on unseen test images.
############################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

PCA_CSV = os.path.join(
    ROOT_DIR,
    "pca_top2_filtered_female_1.csv"
)

ARCFACE_CSV = os.path.join(
    ROOT_DIR,
    "female_arcface_embeddings.csv"
)

OUTPUT_RESULTS = os.path.join(
    ROOT_DIR,
    "facenet_pca2_to_arcface_results.csv"
)

OUTPUT_ANGLE_GRAPH = os.path.join(
    ROOT_DIR,
    "facenet_pca_angle_vs_arcface_prediction.png"
)

OUTPUT_SCATTER = os.path.join(
    ROOT_DIR,
    "facenet_pca2_arcface_linear_fit.png"
)


# ============================================================
# LOAD FACENET PCA
# ============================================================

print("Loading FaceNet PCA...")

pca_df = pd.read_csv(
    PCA_CSV,
    header=None
)

pca_df.columns = [
    "filename",
    "PC1",
    "PC2"
]

pca_df["filename"] = (
    pca_df["filename"]
    .astype(str)
    .apply(os.path.basename)
)

pca_df["angle_deg"] = (
    np.degrees(
        np.arctan2(
            pca_df["PC2"],
            pca_df["PC1"]
        )
    ) % 360
)

pca_df["radius"] = np.sqrt(
    pca_df["PC1"] ** 2
    +
    pca_df["PC2"] ** 2
)


print(
    "PCA images:",
    len(pca_df)
)


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("Loading ArcFace embeddings...")

arc_df = pd.read_csv(
    ARCFACE_CSV,
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


# L2 normalize ArcFace
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


arcface_lookup = {
    name: embedding
    for name, embedding
    in zip(
        arc_names,
        arc_embeddings
    )
}


# ============================================================
# MATCH SAME IMAGES
# ============================================================

common_df = pca_df[
    pca_df["filename"].isin(
        arcface_lookup
    )
].copy()


print(
    "Images existing in both PCA and ArcFace:",
    len(common_df)
)


X = common_df[
    [
        "PC1",
        "PC2"
    ]
].to_numpy(
    dtype=np.float32
)

Y = np.stack(
    [
        arcface_lookup[name]
        for name
        in common_df["filename"]
    ]
)


angles = common_df[
    "angle_deg"
].to_numpy()

filenames = common_df[
    "filename"
].to_numpy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(common_df)
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
    "Train:",
    len(train_idx)
)

print(
    "Test:",
    len(test_idx)
)


# ============================================================
# FIT LINEAR MAPPING:
#
# ArcFace ≈ W * [PC1, PC2] + b
# ============================================================

model = Ridge(
    alpha=1e-6,
    fit_intercept=True
)

model.fit(
    X_train,
    Y_train
)


Y_pred = model.predict(
    X_test
)


# ============================================================
# GLOBAL R²
# ============================================================

global_r2 = r2_score(
    Y_test,
    Y_pred,
    multioutput="variance_weighted"
)

r2_dimensions = r2_score(
    Y_test,
    Y_pred,
    multioutput="raw_values"
)


# ============================================================
# NORMALIZE PREDICTED EMBEDDINGS
# ============================================================

pred_norms = np.linalg.norm(
    Y_pred,
    axis=1,
    keepdims=True
)

pred_norms[
    pred_norms == 0
] = 1

Y_pred_norm = (
    Y_pred / pred_norms
)


true_norms = np.linalg.norm(
    Y_test,
    axis=1,
    keepdims=True
)

true_norms[
    true_norms == 0
] = 1

Y_test_norm = (
    Y_test / true_norms
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

cosine_similarity = np.sum(
    Y_pred_norm
    *
    Y_test_norm,
    axis=1
)


# ============================================================
# RANDOM BASELINE
# ============================================================

rng = np.random.default_rng(
    42
)

random_idx = rng.permutation(
    len(Y_test_norm)
)

random_cosine = np.sum(
    Y_pred_norm
    *
    Y_test_norm[
        random_idx
    ],
    axis=1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("FACENET PCA(2D) -> ARCFACE RESULTS")
print("=" * 70)

print()

print(
    "Global test R²:",
    global_r2
)

print(
    "Mean R² across ArcFace dimensions:",
    np.mean(
        r2_dimensions
    )
)

print(
    "Median R² across ArcFace dimensions:",
    np.median(
        r2_dimensions
    )
)

print()

print(
    "Mean cosine similarity "
    "(predicted vs true ArcFace):",
    np.mean(
        cosine_similarity
    )
)

print(
    "Median cosine similarity:",
    np.median(
        cosine_similarity
    )
)

print(
    "Random baseline cosine:",
    np.mean(
        random_cosine
    )
)

print("=" * 70)


# ============================================================
# SAVE PER-IMAGE RESULTS
# ============================================================

test_results = pd.DataFrame(
    {
        "filename":
            filenames[
                test_idx
            ],

        "PC1":
            X_test[
                :,
                0
            ],

        "PC2":
            X_test[
                :,
                1
            ],

        "angle_deg":
            angles[
                test_idx
            ],

        "cosine_similarity":
            cosine_similarity,

        "random_cosine":
            random_cosine,
    }
)


test_results.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# ============================================================
# ANGULAR ANALYSIS
#
# For each FaceNet PCA angle window:
# how well can PC1/PC2 predict ArcFace?
# ============================================================

window_size = 20

angle_results = []

test_angles = angles[
    test_idx
]


for start_angle in range(
    0,
    360,
    1
):

    end_angle = (
        start_angle
        + window_size
    ) % 360

    if start_angle < end_angle:

        mask = (
            (test_angles >= start_angle)
            &
            (test_angles < end_angle)
        )

    else:

        mask = (
            (test_angles >= start_angle)
            |
            (test_angles < end_angle)
        )


    if np.sum(mask) == 0:

        mean_cosine = np.nan

    else:

        mean_cosine = np.mean(
            cosine_similarity[
                mask
            ]
        )


    center_angle = (
        start_angle
        + window_size / 2
    ) % 360


    angle_results.append(
        (
            center_angle,
            mean_cosine,
            np.sum(mask)
        )
    )


angle_df = pd.DataFrame(
    angle_results,
    columns=[
        "angle",
        "mean_cosine",
        "n_images"
    ]
)

angle_df = angle_df.sort_values(
    "angle"
)


# ============================================================
# PLOT ANGULAR QUALITY
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    angle_df["angle"],
    angle_df["mean_cosine"]
)

plt.xlabel(
    "Angle in FaceNet PCA space"
)

plt.ylabel(
    "Mean cosine similarity\n"
    "Predicted ArcFace vs true ArcFace"
)

plt.title(
    "How Well the FaceNet PC1-PC2 Subspace Predicts ArcFace"
)

plt.xlim(
    0,
    360
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_ANGLE_GRAPH,
    dpi=300
)

plt.close()


# ============================================================
# SIMPLE SCATTER:
# PCA radius vs prediction quality
# ============================================================

test_radius = np.linalg.norm(
    X_test,
    axis=1
)


plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    test_radius,
    cosine_similarity,
    s=10,
    alpha=0.4
)

plt.xlabel(
    "Radius in FaceNet PC1-PC2 space"
)

plt.ylabel(
    "Cosine similarity\n"
    "Predicted ArcFace vs true ArcFace"
)

plt.title(
    "FaceNet PCA Radius vs ArcFace Linear Prediction Quality"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_SCATTER,
    dpi=300
)

plt.close()


print()
print(
    "Saved results:",
    OUTPUT_RESULTS
)

print(
    "Saved angle graph:",
    OUTPUT_ANGLE_GRAPH
)

print(
    "Saved scatter:",
    OUTPUT_SCATTER
)