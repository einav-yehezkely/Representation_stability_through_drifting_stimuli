import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
ARCFACE_CSV = "female_arcface_embeddings.csv"
MODEL_PATH = "model_ft_0_ARCFACE_RESNET50.pth"

OUTPUT_CSV = "pca_points_with_arcface_predictions.csv"
OUTPUT_PLOT = "pca_arcface_predictions.png"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD PCA
# ============================================================

print()
print("=" * 70)
print("LOADING PCA")
print("=" * 70)

pca_df = pd.read_csv(
    PCA_CSV,
    header=None,
    names=["filename", "PC1", "PC2"]
)

print("PCA columns:", list(pca_df.columns))
print("PCA rows:", len(pca_df))


# ------------------------------------------------------------
# Find filename column
# ------------------------------------------------------------

if "filename" in pca_df.columns:
    filename_col = "filename"
else:
    possible = [
        c for c in pca_df.columns
        if "file" in c.lower()
        or "image" in c.lower()
        or "name" in c.lower()
    ]

    if not possible:
        raise RuntimeError(
            "Could not find filename column in PCA CSV."
        )

    filename_col = possible[0]


# ------------------------------------------------------------
# Find PCA columns
# ------------------------------------------------------------

possible_pc1 = [
    c for c in pca_df.columns
    if c.lower() in [
        "pc1",
        "pca1",
        "pca_1"
    ]
]

possible_pc2 = [
    c for c in pca_df.columns
    if c.lower() in [
        "pc2",
        "pca2",
        "pca_2"
    ]
]

if not possible_pc1 or not possible_pc2:
    raise RuntimeError(
        "Could not find PC1 / PC2 columns."
    )

pc1_col = possible_pc1[0]
pc2_col = possible_pc2[0]

print("Filename column:", filename_col)
print("PC1 column:", pc1_col)
print("PC2 column:", pc2_col)


pca_df["filename_clean"] = (
    pca_df[filename_col]
    .astype(str)
    .apply(os.path.basename)
)


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("LOADING ARCFACE EMBEDDINGS")
print("=" * 70)

arc_df = pd.read_csv(
    ARCFACE_CSV,
    header=None
)

print("ArcFace shape:", arc_df.shape)

if arc_df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns, got {arc_df.shape[1]}"
    )


embedding_lookup = {}

for _, row in arc_df.iterrows():

    filename = os.path.basename(
        str(row.iloc[0]).strip()
    )

    embedding = row.iloc[
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    # Same normalization used during training
    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    embedding_lookup[
        filename
    ] = embedding


print(
    "Loaded ArcFace embeddings:",
    len(embedding_lookup)
)


# ============================================================
# CLASSIFIER
#
# MUST MATCH THE ARCHITECTURE USED DURING TRAINING
# ============================================================

class ArcFaceClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# LOAD MODEL
# ============================================================

model = ArcFaceClassifier().to(
    device
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print(
    "Loaded model:",
    MODEL_PATH
)


# ============================================================
# KEEP ONLY IMAGES PRESENT IN BOTH FILES
# ============================================================

valid_df = pca_df[
    pca_df["filename_clean"].isin(
        embedding_lookup
    )
].copy()


print()
print("=" * 70)
print("MATCHING")
print("=" * 70)

print(
    "PCA images:",
    len(pca_df)
)

print(
    "Images with ArcFace embeddings:",
    len(valid_df)
)

print(
    "Missing:",
    len(pca_df) - len(valid_df)
)


# ============================================================
# CLASSIFY ALL POINTS
# ============================================================

BATCH_SIZE = 2048

predictions = []
probabilities_A = []
probabilities_B = []


filenames = valid_df[
    "filename_clean"
].tolist()


print()
print("=" * 70)
print("CLASSIFYING ALL PCA POINTS")
print("=" * 70)


with torch.no_grad():

    for start in range(
        0,
        len(filenames),
        BATCH_SIZE
    ):

        batch_names = filenames[
            start:start + BATCH_SIZE
        ]

        batch_embeddings = np.stack([
            embedding_lookup[name]
            for name in batch_names
        ])


        x = torch.tensor(
            batch_embeddings,
            dtype=torch.float32,
            device=device
        )


        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )


        predictions.extend(
            preds.cpu().numpy().tolist()
        )

        probabilities_A.extend(
            probs[:, 0]
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities_B.extend(
            probs[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )


valid_df["prediction"] = predictions

valid_df["predicted_class"] = np.where(
    valid_df["prediction"] == 0,
    "A",
    "B"
)

valid_df["P_A"] = probabilities_A
valid_df["P_B"] = probabilities_B


# ============================================================
# SUMMARY
# ============================================================

num_A = (
    valid_df["predicted_class"] == "A"
).sum()

num_B = (
    valid_df["predicted_class"] == "B"
).sum()


print()
print("=" * 70)
print("PREDICTIONS")
print("=" * 70)

print(
    f"Predicted A: {num_A} "
    f"({100 * num_A / len(valid_df):.2f}%)"
)

print(
    f"Predicted B: {num_B} "
    f"({100 * num_B / len(valid_df):.2f}%)"
)


# ============================================================
# SAVE DATA
# ============================================================

valid_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    "Saved predictions:",
    OUTPUT_CSV
)


# ============================================================
# PLOT
# ============================================================

A_points = valid_df[
    valid_df["predicted_class"] == "A"
]

B_points = valid_df[
    valid_df["predicted_class"] == "B"
]


plt.figure(
    figsize=(10, 10)
)


# ------------------------------------------------------------
# A = BLUE
# ------------------------------------------------------------

plt.scatter(
    A_points[pc1_col],
    A_points[pc2_col],
    s=3,
    c="blue",
    alpha=0.45,
    label=f"Predicted A ({len(A_points)})"
)


# ------------------------------------------------------------
# B = RED
# ------------------------------------------------------------

plt.scatter(
    B_points[pc1_col],
    B_points[pc2_col],
    s=3,
    c="red",
    alpha=0.45,
    label=f"Predicted B ({len(B_points)})"
)


plt.xlabel(
    "FaceNet PCA – PC1"
)

plt.ylabel(
    "FaceNet PCA – PC2"
)

plt.title(
    "ArcFace Classifier Predictions in FaceNet PCA Space"
)

plt.axis(
    "equal"
)

plt.legend(
    markerscale=4
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=300
)

plt.show()


print()
print(
    "Saved plot:",
    OUTPUT_PLOT
)