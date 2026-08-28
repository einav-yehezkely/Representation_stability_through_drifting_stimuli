import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
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

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "model_ft_0_CE_FACENET.pth"
)

EMBEDDINGS_CSV = os.path.join(
    ROOT_DIR,
    "female_facenet_embeddings.csv"
)

PCA_CSV = os.path.join(
    ROOT_DIR,
    "pca_top2_filtered_female_1.csv"
)

ROTATION_SEQUENCE_CSV = os.path.join(
    ROOT_DIR,
    "rotation_sequence_all.csv"
)

OUTPUT_GRAPH = os.path.join(
    ROOT_DIR,
    "initial_facenet_classifier_graph.png"
)

OUTPUT_PREDICTIONS = os.path.join(
    ROOT_DIR,
    "initial_facenet_predictions.csv"
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# FACENET CLASSIFIER
# ============================================================
#
# IMPORTANT:
# This must match the architecture used when
# model_ft_0_CE_FACENET.pth was trained.
#
# FaceNet embedding:
# 512 -> 64 -> 2
#
# ============================================================

class FaceNetClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

files_to_check = {
    "Model": MODEL_PATH,
    "Embeddings": EMBEDDINGS_CSV,
    "PCA": PCA_CSV,
    "Rotation": ROTATION_SEQUENCE_CSV,
}

for name, path in files_to_check.items():

    print(
        f"{name}:",
        path
    )

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )


# ============================================================
# LOAD PCA
# ============================================================

print()
print("=" * 70)
print("LOADING PCA")
print("=" * 70)

pca_df = pd.read_csv(
    PCA_CSV,
    header=None
)

pca_df = pca_df.iloc[
    :,
    :3
].copy()

pca_df.columns = [
    "filename",
    "x",
    "y"
]

pca_df["filename"] = (
    pca_df["filename"]
    .astype(str)
    .apply(os.path.basename)
)

pca_df["x"] = pd.to_numeric(
    pca_df["x"],
    errors="coerce"
)

pca_df["y"] = pd.to_numeric(
    pca_df["y"],
    errors="coerce"
)

pca_df = pca_df.dropna(
    subset=[
        "x",
        "y"
    ]
)

pca_df["angle_deg"] = (
    np.degrees(
        np.arctan2(
            pca_df["y"],
            pca_df["x"]
        )
    )
    % 360
)

ANGLE_MAP = dict(
    zip(
        pca_df["filename"],
        pca_df["angle_deg"]
    )
)

print(
    "PCA images:",
    len(ANGLE_MAP)
)


# ============================================================
# LOAD FACENET EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("LOADING FACENET EMBEDDINGS")
print("=" * 70)

embeddings_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None
)

embedding_lookup = {}

for _, row in embeddings_df.iterrows():

    filename = str(
        row.iloc[0]
    ).strip()

    basename = os.path.basename(
        filename
    )

    embedding = row.iloc[
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    if len(embedding) != 512:

        continue

    if not np.all(
        np.isfinite(embedding)
    ):

        continue

    # --------------------------------------------------------
    # L2 normalization
    # --------------------------------------------------------

    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:

        embedding = (
            embedding
            / norm
        )

    embedding_lookup[
        basename
    ] = embedding


print(
    "FaceNet embeddings loaded:",
    len(embedding_lookup)
)


# ============================================================
# LOAD INITIAL MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING FACENET CLASSIFIER")
print("=" * 70)

model = FaceNetClassifier().to(
    device
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)


# ------------------------------------------------------------
# Sometimes a checkpoint contains:
#
# {
#     "model_state_dict": ...
# }
#
# instead of directly containing the weights.
# ------------------------------------------------------------

if isinstance(
    state_dict,
    dict
):

    if "model_state_dict" in state_dict:

        state_dict = state_dict[
            "model_state_dict"
        ]

    elif "state_dict" in state_dict:

        state_dict = state_dict[
            "state_dict"
        ]


# ------------------------------------------------------------
# Remove "module." prefix if model was trained with DataParallel
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith(
        "module."
    ):

        new_key = new_key[
            len("module.") :
        ]

    clean_state_dict[
        new_key
    ] = value


# ------------------------------------------------------------
# Try loading
# ------------------------------------------------------------

try:

    model.load_state_dict(
        clean_state_dict
    )

except RuntimeError as e:

    print()
    print("=" * 70)
    print("MODEL ARCHITECTURE DOES NOT MATCH")
    print("=" * 70)

    print(
        "\nThe saved model does not match:"
    )

    print(
        "512 -> 64 -> 2"
    )

    print()
    print(
        "Saved model keys:"
    )

    for key in clean_state_dict.keys():

        print(
            " ",
            key,
            tuple(
                clean_state_dict[
                    key
                ].shape
            )
        )

    raise e


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD ROTATION SEQUENCE
# ============================================================

print()
print("=" * 70)
print("LOADING ROTATION SEQUENCE")
print("=" * 70)

rotation_df = pd.read_csv(
    ROTATION_SEQUENCE_CSV
)

if "filename" not in rotation_df.columns:

    raise ValueError(
        "rotation_sequence_all.csv "
        "does not contain a 'filename' column."
    )

rotation_df["filename"] = (
    rotation_df["filename"]
    .astype(str)
)

print(
    "Rotation sequence rows:",
    len(rotation_df)
)


# ============================================================
# CLASSIFY — NO TRAINING
# ============================================================

print()
print("=" * 70)
print("CLASSIFYING ROTATION IMAGES")
print("=" * 70)

records = []

missing_embeddings = 0

with torch.no_grad():

    for _, row in rotation_df.iterrows():

        filename = str(
            row["filename"]
        )

        basename = os.path.basename(
            filename
        )

        if basename not in embedding_lookup:

            missing_embeddings += 1

            print(
                "Missing embedding:",
                basename
            )

            continue

        embedding_np = embedding_lookup[
            basename
        ]

        embedding = torch.tensor(
            embedding_np.tolist(),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        logits = model(
            embedding
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        records.append(
            {
                "filename": basename,
                "pred": (
                    "A"
                    if pred == 0
                    else "B"
                ),
                "prob_A": probs[
                    0,
                    0
                ].item(),
                "prob_B": probs[
                    0,
                    1
                ].item(),
            }
        )


df = pd.DataFrame(
    records
)

print()
print(
    "Classified:",
    len(df)
)

print(
    "Missing embeddings:",
    missing_embeddings
)


# ============================================================
# ADD PCA ANGLE
# ============================================================

df["angle_deg"] = df[
    "filename"
].map(
    ANGLE_MAP
)

missing_angles = df[
    "angle_deg"
].isna().sum()

print(
    "Missing PCA angles:",
    missing_angles
)

df = df.dropna(
    subset=[
        "angle_deg"
    ]
).copy()


# ============================================================
# BASIC CLASSIFICATION SUMMARY
# ============================================================

print()
print("=" * 70)
print("INITIAL CLASSIFICATION SUMMARY")
print("=" * 70)

count_a = (
    df["pred"]
    == "A"
).sum()

count_b = (
    df["pred"]
    == "B"
).sum()

total = len(
    df
)

if total > 0:

    print(
        f"A: {count_a} "
        f"({100 * count_a / total:.2f}%)"
    )

    print(
        f"B: {count_b} "
        f"({100 * count_b / total:.2f}%)"
    )


# ============================================================
# COMPUTE A/B PERCENTAGES IN 20-DEGREE WINDOWS
# ============================================================

window_size = 20

results = []

for start_angle in range(
    0,
    360
):

    end_angle = (
        start_angle
        + window_size
    ) % 360

    # --------------------------------------------------------
    # Normal window
    # --------------------------------------------------------

    if start_angle < end_angle:

        window_data = df[
            (
                df["angle_deg"]
                >= start_angle
            )
            &
            (
                df["angle_deg"]
                < end_angle
            )
        ]

    # --------------------------------------------------------
    # Window crosses 360 -> 0
    # --------------------------------------------------------

    else:

        window_data = df[
            (
                df["angle_deg"]
                >= start_angle
            )
            |
            (
                df["angle_deg"]
                < end_angle
            )
        ]

    n = len(
        window_data
    )

    if n > 0:

        n_a = (
            window_data[
                "pred"
            ]
            == "A"
        ).sum()

        percent_a = (
            100.0
            * n_a
            / n
        )

        percent_b = (
            100.0
            - percent_a
        )

        mean_prob_a = (
            window_data[
                "prob_A"
            ].mean()
            * 100
        )

    else:

        percent_a = np.nan
        percent_b = np.nan
        mean_prob_a = np.nan

    center_angle = (
        start_angle
        + window_size / 2
    ) % 360

    results.append(
        {
            "angle": center_angle,
            "percent_A": percent_a,
            "percent_B": percent_b,
            "mean_prob_A": mean_prob_a,
            "n_images": n,
        }
    )


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "angle"
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(
        12,
        6
    )
)

plt.plot(
    results_df[
        "angle"
    ],
    results_df[
        "percent_A"
    ],
    label="Predicted A",
    linewidth=2
)

plt.plot(
    results_df[
        "angle"
    ],
    results_df[
        "percent_B"
    ],
    label="Predicted B",
    linewidth=2
)

# plt.axhline(
#     y=50,
#     linestyle="--",
#     linewidth=1
# )

plt.xlabel(
    "Angle in FaceNet PCA space"
)

plt.ylabel(
    "% of images classified as A/B"
)

plt.title(
    "Initial FaceNet Classifier Before Self-Training"
)

plt.ylim(
    0,
    100
)

plt.xlim(
    0,
    360
)

plt.xticks(
    np.arange(
        0,
        361,
        30
    )
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_GRAPH,
    dpi=300
)

plt.close()


# ============================================================
# SAVE RAW CLASSIFICATIONS
# ============================================================

df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# SAVE WINDOW RESULTS
# ============================================================

WINDOW_OUTPUT = os.path.join(
    ROOT_DIR,
    "initial_facenet_predictions_20deg_windows.csv"
)

results_df.to_csv(
    WINDOW_OUTPUT,
    index=False
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved graph:"
)

print(
    OUTPUT_GRAPH
)

print()

print(
    "Saved predictions:"
)

print(
    OUTPUT_PREDICTIONS
)

print()

print(
    "Saved window results:"
)

print(
    WINDOW_OUTPUT
)