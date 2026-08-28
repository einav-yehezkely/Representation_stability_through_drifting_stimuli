############################################################
# TEST LINEAR TRANSFORMATION:
#
# FaceNet 512D  --->  Linear transformation  ---> ArcFace 512D
#
# Tests whether ArcFace representations can be predicted
# from FaceNet representations using a single linear mapping.
#
# IMPORTANT:
# The linear transformation is fitted ONLY on the training set
# and evaluated on unseen test images.
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

FACENET_CSV = os.path.join(
    ROOT_DIR,
    "female_facenet_embeddings.csv"
)

ARCFACE_CSV = os.path.join(
    ROOT_DIR,
    "female_arcface_embeddings.csv"
)

OUTPUT_CSV = os.path.join(
    ROOT_DIR,
    "facenet_to_arcface_linear_mapping_results.csv"
)

OUTPUT_GRAPH = os.path.join(
    ROOT_DIR,
    "facenet_to_arcface_linear_mapping.png"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(csv_path):

    df = pd.read_csv(
        csv_path,
        header=None
    )

    filenames = df.iloc[:, 0].astype(str).apply(
        os.path.basename
    )

    embeddings = df.iloc[
        :,
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    # ---------------------------------------------
    # L2 normalize every embedding
    # ---------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[
        norms == 0
    ] = 1

    embeddings = (
        embeddings / norms
    )

    return filenames, embeddings


print("Loading FaceNet...")

facenet_names, facenet_embeddings = load_embeddings(
    FACENET_CSV
)

print(
    "FaceNet:",
    len(facenet_names)
)


print("Loading ArcFace...")

arcface_names, arcface_embeddings = load_embeddings(
    ARCFACE_CSV
)

print(
    "ArcFace:",
    len(arcface_names)
)


# ============================================================
# MATCH SAME IMAGES BETWEEN NETWORKS
# ============================================================

facenet_dict = {
    name: embedding
    for name, embedding
    in zip(
        facenet_names,
        facenet_embeddings
    )
}

arcface_dict = {
    name: embedding
    for name, embedding
    in zip(
        arcface_names,
        arcface_embeddings
    )
}


common_names = sorted(
    set(facenet_dict.keys())
    &
    set(arcface_dict.keys())
)


print()
print(
    "Images existing in BOTH networks:",
    len(common_names)
)


# ============================================================
# BUILD MATRICES
#
# X = FaceNet
# Y = ArcFace
# ============================================================

X = np.stack(
    [
        facenet_dict[name]
        for name in common_names
    ]
)

Y = np.stack(
    [
        arcface_dict[name]
        for name in common_names
    ]
)


print(
    "X shape:",
    X.shape
)

print(
    "Y shape:",
    Y.shape
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = np.arange(
    len(common_names)
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
    "Training images:",
    len(train_idx)
)

print(
    "Test images:",
    len(test_idx)
)


# ============================================================
# FIT LINEAR TRANSFORMATION
#
# ArcFace ≈ FaceNet @ W + b
#
# Ridge with tiny regularization is used for numerical
# stability. This is still a linear/affine transformation.
# ============================================================

print()
print(
    "Fitting linear transformation..."
)


model = Ridge(
    alpha=1e-6,
    fit_intercept=True
)

model.fit(
    X_train,
    Y_train
)


# ============================================================
# PREDICT ARCFACE EMBEDDINGS
# ============================================================

Y_pred = model.predict(
    X_test
)


# ============================================================
# GLOBAL R²
# ============================================================

r2 = r2_score(
    Y_test,
    Y_pred,
    multioutput="variance_weighted"
)


# ============================================================
# R² FOR EACH ARCFACE DIMENSION
# ============================================================

r2_dimensions = r2_score(
    Y_test,
    Y_pred,
    multioutput="raw_values"
)


# ============================================================
# COSINE SIMILARITY:
#
# predicted ArcFace vs real ArcFace
# ============================================================

pred_norms = np.linalg.norm(
    Y_pred,
    axis=1,
    keepdims=True
)

pred_norms[
    pred_norms == 0
] = 1

Y_pred_normalized = (
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

Y_test_normalized = (
    Y_test / true_norms
)


cosine_similarity = np.sum(
    Y_pred_normalized
    *
    Y_test_normalized,
    axis=1
)


# ============================================================
# EUCLIDEAN ERROR
# ============================================================

euclidean_error = np.linalg.norm(
    Y_pred_normalized
    -
    Y_test_normalized,
    axis=1
)


# ============================================================
# BASELINE
#
# Compare predicted ArcFace embedding with a RANDOM
# ArcFace embedding.
# ============================================================

rng = np.random.default_rng(
    42
)

random_indices = rng.permutation(
    len(Y_test_normalized)
)

random_cosine = np.sum(
    Y_pred_normalized
    *
    Y_test_normalized[
        random_indices
    ],
    axis=1
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("LINEAR TRANSFORMATION RESULTS")
print("=" * 70)

print()

print(
    "Global test R²:",
    r2
)

print()

print(
    "Mean R² across 512 dimensions:",
    np.mean(r2_dimensions)
)

print(
    "Median R² across 512 dimensions:",
    np.median(r2_dimensions)
)

print()

print(
    "Mean cosine similarity "
    "(predicted vs true ArcFace):",
    np.mean(cosine_similarity)
)

print(
    "Median cosine similarity:",
    np.median(cosine_similarity)
)

print()

print(
    "Mean cosine similarity "
    "(predicted vs RANDOM ArcFace):",
    np.mean(random_cosine)
)

print()

print(
    "Mean Euclidean error:",
    np.mean(euclidean_error)
)

print("=" * 70)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

results_df = pd.DataFrame(
    {
        "filename":
            [
                common_names[i]
                for i in test_idx
            ],

        "cosine_similarity":
            cosine_similarity,

        "euclidean_error":
            euclidean_error,

        "random_cosine_similarity":
            random_cosine,
    }
)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PLOT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    cosine_similarity,
    bins=50,
    alpha=0.7,
    label="Correct ArcFace embedding"
)

plt.hist(
    random_cosine,
    bins=50,
    alpha=0.7,
    label="Random ArcFace embedding"
)

plt.xlabel(
    "Cosine similarity"
)

plt.ylabel(
    "Number of test images"
)

plt.title(
    "FaceNet → Linear Mapping → ArcFace"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_GRAPH,
    dpi=300
)

plt.close()


print()
print(
    "Saved results:",
    OUTPUT_CSV
)

print(
    "Saved graph:",
    OUTPUT_GRAPH
)