import os
import sys

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

sys.path.append(
    PROJECT_DIR
)


# ============================================================
# IMPORT NEW CLASSIFIER
# ============================================================

from resnet50_embeddings_training import ResNet50Classifier


# ============================================================
# SETTINGS
# ============================================================

EMBEDDING_DIM = 2048


MODEL_PATH = os.path.join(
    ROOT_DIR,
    "model_ft_0_RESNET50_VGGFACE2.pth"
)


EMBEDDINGS_CSV = os.path.join(
    ROOT_DIR,
    "female_resnet50_vggface2_embeddings.csv"
)


# FaceNet PCA is still used to define geometry / angle
PCA_CSV = os.path.join(
    ROOT_DIR,
    "pca_top2_filtered_female_vgg_1.csv"
)


ROTATION_SEQUENCE_CSV = os.path.join(
    ROOT_DIR,
    "rotation_sequence_all.csv"
)


OUTPUT_PATH = os.path.join(
    ROOT_DIR,
    "initial_resnet50_vggface2_linear_graph.png"
)


PREDICTIONS_CSV = os.path.join(
    ROOT_DIR,
    "initial_resnet50_vggface2_predictions.csv"
)


WINDOW_RESULTS_CSV = os.path.join(
    ROOT_DIR,
    "initial_resnet50_vggface2_window_results.csv"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Using device:",
    device
)


# ============================================================
# LOAD PCA ANGLES
#
# IMPORTANT:
#
# FaceNet is used ONLY for the PCA geometry.
#
# PCA:
#   filename -> angle
#
# Classification:
#   filename -> ResNet50 embedding -> classifier
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING FACENET PCA GEOMETRY"
)

print(
    "=" * 70
)


pca_df = pd.read_csv(
    PCA_CSV,
    header=None
)


pca_df.columns = [
    "filename",
    "x",
    "y"
]


# Normalize filename representation
pca_df["filename"] = (
    pca_df["filename"]
    .astype(str)
    .str.strip()
    .apply(os.path.basename)
)


# ============================================================
# COMPUTE ANGLE IN FACENET PCA SPACE
# ============================================================

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

        pca_df[
            "filename"
        ],

        pca_df[
            "angle_deg"
        ]

    )

)


print(
    "PCA rows:",
    len(
        pca_df
    )
)


print(
    "Example PCA filename:",
    pca_df.iloc[
        0
    ]["filename"]
)


print(
    "Example PCA angle:",
    pca_df.iloc[
        0
    ]["angle_deg"]
)


# ============================================================
# LOAD RESNET50 VGGFACE2 EMBEDDINGS
#
# CSV:
#
# filename, feature_1, ..., feature_2048
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING RESNET50 VGGFACE2 EMBEDDINGS"
)

print(
    "=" * 70
)


embeddings_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None
)


expected_columns = (
    1
    + EMBEDDING_DIM
)


if embeddings_df.shape[1] != expected_columns:

    raise RuntimeError(

        f"Expected {expected_columns} columns "
        f"(1 filename + {EMBEDDING_DIM} embedding values), "
        f"but found {embeddings_df.shape[1]}"

    )


print(
    "CSV shape:",
    embeddings_df.shape
)


print(
    "Embedding dimension:",
    embeddings_df.shape[1] - 1
)


# ============================================================
# BUILD EMBEDDING LOOKUP
#
# filename -> normalized 2048D embedding
# ============================================================

embedding_lookup = {}


for _, row in embeddings_df.iterrows():

    filename = str(
        row.iloc[0]
    ).strip()


    basename = os.path.basename(
        filename
    )


    embedding = row.iloc[
        1:1 + EMBEDDING_DIM
    ].to_numpy(
        dtype=np.float32
    )


    # --------------------------------------------------------
    # L2 NORMALIZATION
    #
    # Must be the same normalization used during training.
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
    "Embeddings loaded:",
    len(
        embedding_lookup
    )
)


example_filename = next(
    iter(
        embedding_lookup
    )
)


print(
    "Example filename:",
    example_filename
)


print(
    "Example embedding shape:",
    embedding_lookup[
        example_filename
    ].shape
)


# ============================================================
# LOAD TRAINED RESNET50 CLASSIFIER
#
# Architecture:
#
# 2048 -> 64 -> ReLU -> 2
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING CLASSIFIER"
)

print(
    "=" * 70
)


model = ResNet50Classifier()


state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    state_dict
)


model = model.to(
    device
)


model.eval()


print(
    "Loaded model:",
    MODEL_PATH
)


print(
    "Classifier:"
)

print(
    model
)


# ============================================================
# LOAD ROTATION SEQUENCE
# ============================================================

print()

print(
    "=" * 70
)

print(
    "LOADING ROTATION SEQUENCE"
)

print(
    "=" * 70
)


rotation_df = pd.read_csv(
    ROTATION_SEQUENCE_CSV
)


if "filename" not in rotation_df.columns:

    raise RuntimeError(

        "rotation_sequence_all.csv "
        "does not contain a 'filename' column."

    )


print(
    "Rotation sequence rows:",
    len(
        rotation_df
    )
)


# ============================================================
# CLASSIFY ROTATION SEQUENCE
#
# NO TRAINING
#
# FaceNet PCA-selected filename
#           ↓
# ResNet50 VGGFace2 2048D embedding
#           ↓
# classifier
#           ↓
# A / B
# ============================================================

print()

print(
    "=" * 70
)

print(
    "CLASSIFYING ROTATION SEQUENCE"
)

print(
    "=" * 70
)


records = []


missing_embeddings = []


with torch.no_grad():

    for _, row in rotation_df.iterrows():

        filename = str(
            row["filename"]
        ).strip()


        basename = os.path.basename(
            filename
        )


        # ----------------------------------------------------
        # CHECK EMBEDDING
        # ----------------------------------------------------

        if basename not in embedding_lookup:

            missing_embeddings.append(
                basename
            )

            continue


        # ----------------------------------------------------
        # GET 2048D EMBEDDING
        # ----------------------------------------------------

        embedding_np = embedding_lookup[
            basename
        ]


        embedding = torch.tensor(

            embedding_np,

            dtype=torch.float32

        ).unsqueeze(
            0
        ).to(
            device
        )


        # ----------------------------------------------------
        # MODEL OUTPUT
        # ----------------------------------------------------

        output = model(
            embedding
        )


        probs = torch.softmax(
            output,
            dim=1
        )


        pred = output.argmax(
            dim=1
        ).item()


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        records.append(

            {

                "filename":
                    basename,

                "pred":
                    "A"
                    if pred == 0
                    else "B",

                "prob_A":
                    probs[
                        0,
                        0
                    ].item(),

                "prob_B":
                    probs[
                        0,
                        1
                    ].item(),

            }

        )


print(
    "Successfully classified:",
    len(
        records
    )
)


print(
    "Missing embeddings:",
    len(
        missing_embeddings
    )
)


if missing_embeddings:

    print(
        "Examples:",
        missing_embeddings[:10]
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    records
)


if len(df) == 0:

    raise RuntimeError(
        "No images were classified."
    )


# ============================================================
# ADD FACENET PCA ANGLE
# ============================================================

df["angle_deg"] = df[
    "filename"
].map(
    ANGLE_MAP
)


missing_angles = df[
    "angle_deg"
].isna()


print()

print(
    "Missing PCA angles:",
    missing_angles.sum()
)


if missing_angles.any():

    print(
        "Examples:",
        df.loc[
            missing_angles,
            "filename"
        ].head(
            10
        ).tolist()
    )


df = df.dropna(

    subset=[
        "angle_deg"
    ]

).copy()


# ============================================================
# SORT BY PCA ANGLE
# ============================================================

df = df.sort_values(
    "angle_deg"
).reset_index(
    drop=True
)


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

df.to_csv(
    PREDICTIONS_CSV,
    index=False
)


print()

print(
    "Saved raw predictions:",
    PREDICTIONS_CSV
)


# ============================================================
# OVERALL PREDICTION COUNTS
# ============================================================

count_a = (
    df["pred"]
    == "A"
).sum()


count_b = (
    df["pred"]
    == "B"
).sum()


total_predictions = len(
    df
)


print()

print(
    "=" * 70
)

print(
    "OVERALL PREDICTIONS"
)

print(
    "=" * 70
)


print(
    f"A: {count_a} "
    f"({100 * count_a / total_predictions:.2f}%)"
)


print(
    f"B: {count_b} "
    f"({100 * count_b / total_predictions:.2f}%)"
)


# ============================================================
# COMPUTE A/B PERCENTAGES IN 20 DEGREE WINDOWS
#
# Window example:
#
# 0° -> 20°
# 1° -> 21°
# 2° -> 22°
# ...
#
# Result plotted at center of window.
# ============================================================

WINDOW_SIZE = 20


results = []


for start_angle in range(
    0,
    360
):

    end_angle = (
        start_angle
        + WINDOW_SIZE
    ) % 360


    # --------------------------------------------------------
    # NORMAL WINDOW
    #
    # Example:
    # 40 -> 60
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
    # WRAP-AROUND WINDOW
    #
    # Example:
    # 350 -> 10
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


    total = len(
        window_data
    )


    if total > 0:

        window_count_a = (

            window_data[
                "pred"
            ]

            == "A"

        ).sum()


        window_count_b = (

            window_data[
                "pred"
            ]

            == "B"

        ).sum()


        percent_a = (

            100.0

            * window_count_a

            / total

        )


        percent_b = (

            100.0

            * window_count_b

            / total

        )


        mean_prob_a = (
            window_data[
                "prob_A"
            ].mean()
        )


        mean_prob_b = (
            window_data[
                "prob_B"
            ].mean()
        )


    else:

        percent_a = np.nan

        percent_b = np.nan

        mean_prob_a = np.nan

        mean_prob_b = np.nan


    center_angle = (

        start_angle

        + WINDOW_SIZE / 2

    ) % 360


    results.append(

        {

            "angle":
                center_angle,

            "percent_A":
                percent_a,

            "percent_B":
                percent_b,

            "mean_prob_A":
                mean_prob_a,

            "mean_prob_B":
                mean_prob_b,

            "num_images":
                total,

        }

    )


# ============================================================
# CREATE WINDOW DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "angle"
).reset_index(
    drop=True
)


# ============================================================
# SAVE WINDOW RESULTS
# ============================================================

results_df.to_csv(
    WINDOW_RESULTS_CSV,
    index=False
)


print()

print(
    "Saved window results:",
    WINDOW_RESULTS_CSV
)


# ============================================================
# MAIN GRAPH
#
# Percentage of HARD A/B predictions
# in 20-degree FaceNet PCA windows.
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

    linewidth=2,

)


plt.plot(

    results_df[
        "angle"
    ],

    results_df[
        "percent_B"
    ],

    label="Predicted B",

    linewidth=2,

)


plt.axhline(

    y=50,

    linestyle="--",

    alpha=0.5,

    label="50%"

)


plt.xlabel(
    "Angle in FaceNet PCA space"
)


plt.ylabel(
    "% of images"
)


plt.title(
    "Initial VGGFace2 ResNet50 Classifier Before Self-Training"
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
    OUTPUT_PATH,
    dpi=300
)


plt.close()


print()

print(
    "Saved graph:",
    OUTPUT_PATH
)


# ============================================================
# OPTIONAL:
# MEAN SOFTMAX PROBABILITY GRAPH
# ============================================================

PROBABILITY_OUTPUT_PATH = os.path.join(

    ROOT_DIR,

    "initial_resnet50_vggface2_mean_probability.png"

)


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
        "mean_prob_A"
    ] * 100,

    label="Mean P(A)",

    linewidth=2,

)


plt.plot(

    results_df[
        "angle"
    ],

    results_df[
        "mean_prob_B"
    ] * 100,

    label="Mean P(B)",

    linewidth=2,

)


plt.axhline(

    y=50,

    linestyle="--",

    alpha=0.5,

    label="Decision boundary",

)


plt.xlabel(
    "Angle in FaceNet PCA space"
)


plt.ylabel(
    "Mean classifier probability (%)"
)


plt.title(
    "VGGFace2 ResNet50 Classification Probability Along FaceNet PCA"
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
    PROBABILITY_OUTPUT_PATH,
    dpi=300
)


plt.close()


print(
    "Saved probability graph:",
    PROBABILITY_OUTPUT_PATH
)


# ============================================================
# RAW P(A) BY IMAGE
# ============================================================

RAW_PROBABILITY_OUTPUT_PATH = os.path.join(

    ROOT_DIR,

    "initial_resnet50_vggface2_raw_probability.png"

)


plt.figure(
    figsize=(
        12,
        6
    )
)


plt.scatter(

    df[
        "angle_deg"
    ],

    df[
        "prob_A"
    ] * 100,

    s=12,

    alpha=0.5,

)


plt.axhline(

    y=50,

    linestyle="--",

    alpha=0.5,

    label="Decision boundary",

)


plt.xlabel(
    "Angle in FaceNet PCA space"
)


plt.ylabel(
    "P(A) (%)"
)


plt.title(
    "Raw VGGFace2 ResNet50 Predictions Along FaceNet PCA"
)


plt.xlim(
    0,
    360
)


plt.ylim(
    0,
    100
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
    RAW_PROBABILITY_OUTPUT_PATH,
    dpi=300
)


plt.close()


print(
    "Saved raw probability graph:",
    RAW_PROBABILITY_OUTPUT_PATH
)


# ============================================================
# FINISHED
# ============================================================

print()

print(
    "=" * 70
)

print(
    "DONE"
)

print(
    "=" * 70
)


print(
    "PCA geometry: FaceNet"
)


print(
    "Classifier representation: VGGFace2 ResNet50"
)


print(
    f"Embedding dimension: {EMBEDDING_DIM}"
)


print(
    "Classifier: 2048 -> 64 -> ReLU -> 2"
)


print(
    "No self-training was performed."
)


print(
    "=" * 70
)