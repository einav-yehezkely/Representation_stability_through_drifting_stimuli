import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

sys.path.append(
    PROJECT_DIR
)

from arcface_embeddings_training import ArcFaceClassifier


# ============================================================
# SETTINGS
# ============================================================

ROOT_DIR = os.path.dirname(PROJECT_DIR)

MODEL_PATH = os.path.join(
    ROOT_DIR,
    "model_ft_0_ARCFACE_RESNET50_M.pth"
)

EMBEDDINGS_CSV = os.path.join(
    ROOT_DIR,
    "female_arcface_embeddings.csv"
)

PCA_CSV = os.path.join(
    ROOT_DIR,
    "pca_top2_filtered_female_1.csv"
)

ROTATION_SEQUENCE_CSV = os.path.join(
    ROOT_DIR,
    "rotation_sequence_all.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT_DIR,
    "initial_arcface_linear_graph_perceptron.png"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD PCA ANGLES
# ============================================================

pca_df = pd.read_csv(
    PCA_CSV,
    header=None
)

pca_df.columns = [
    "filename",
    "x",
    "y"
]

pca_df["angle_deg"] = (
    np.degrees(
        np.arctan2(
            pca_df["y"],
            pca_df["x"]
        )
    ) % 360
)

ANGLE_MAP = dict(
    zip(
        pca_df["filename"],
        pca_df["angle_deg"]
    )
)


# ============================================================
# LOAD + NORMALIZE ARCFACE EMBEDDINGS
# ============================================================

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

    # L2 normalization
    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    embedding_lookup[
        basename
    ] = embedding


# ============================================================
# LOAD INITIAL MODEL
# ============================================================

model = ArcFaceClassifier()

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


# ============================================================
# LOAD ROTATION SEQUENCE
# ============================================================

rotation_df = pd.read_csv(
    ROTATION_SEQUENCE_CSV
)

records = []


# ============================================================
# CLASSIFY — NO TRAINING
# ============================================================

with torch.no_grad():

    for _, row in rotation_df.iterrows():

        filename = str(
            row["filename"]
        )

        basename = os.path.basename(
            filename
        )

        if basename not in embedding_lookup:
            print(
                "Missing embedding:",
                filename
            )
            continue

        embedding = torch.tensor(
            embedding_lookup[
                basename
            ],
            dtype=torch.float32
        ).unsqueeze(0).to(
            device
        )

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

        records.append(
            {
                "filename": filename,
                "pred": "A" if pred == 0 else "B",
                "prob_A": probs[0, 0].item(),
                "prob_B": probs[0, 1].item(),
            }
        )


df = pd.DataFrame(
    records
)


# ============================================================
# ADD PCA ANGLE
# ============================================================

df["angle_deg"] = df[
    "filename"
].map(
    ANGLE_MAP
)

df = df.dropna(
    subset=[
        "angle_deg"
    ]
)


# ============================================================
# COMPUTE A/B PERCENTAGES IN 20° WINDOWS
# ============================================================

window_size = 20

results = []

for step_angle in range(
    0,
    360,
    1
):

    end = (
        step_angle
        + window_size
    ) % 360

    if step_angle < end:

        window_data = df[
            (df["angle_deg"] >= step_angle)
            &
            (df["angle_deg"] < end)
        ]

    else:

        window_data = df[
            (df["angle_deg"] >= step_angle)
            |
            (df["angle_deg"] < end)
        ]

    total = len(
        window_data
    )

    if total > 0:

        count_a = (
            window_data["pred"]
            == "A"
        ).sum()

        percent_a = (
            100
            * count_a
            / total
        )

        percent_b = (
            100
            - percent_a
        )

    else:

        percent_a = np.nan
        percent_b = np.nan

    center_angle = (
        step_angle
        + window_size / 2
    ) % 360

    results.append(
        (
            center_angle,
            percent_a,
            percent_b
        )
    )


results_df = pd.DataFrame(
    results,
    columns=[
        "angle",
        "percent_A",
        "percent_B"
    ]
)


# ============================================================
# PLOT
# ============================================================

results_df = results_df.sort_values(
    "angle"
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    results_df["angle"],
    results_df["percent_A"],
    label="Predicted A"
)

plt.plot(
    results_df["angle"],
    results_df["percent_B"],
    label="Predicted B"
)

# plt.axvline(
#     x=0,
#     linestyle="--"
# )

# plt.axvline(
#     x=180,
#     linestyle="--"
# )

plt.xlabel(
    "Angle in FaceNet PCA space"
)

plt.ylabel(
    "% of images"
)

plt.title(
    "Initial ArcFace Classifier Before Self-Training"
)

plt.ylim(
    0,
    100
)

plt.xlim(
    0,
    360
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.close()

# # ============================================================
# # SORT BY PCA ANGLE
# # ============================================================

# df = df.sort_values(
#     "angle_deg"
# ).reset_index(
#     drop=True
# )


# # ============================================================
# # PLOT RAW SOFTMAX PROBABILITIES
# # ============================================================

# plt.figure(
#     figsize=(12, 6)
# )

# plt.plot(
#     df["angle_deg"],
#     df["prob_A"] * 100,
#     label="P(A)",
#     linewidth=1.5
# )

# plt.plot(
#     df["angle_deg"],
#     df["prob_B"] * 100,
#     label="P(B)",
#     linewidth=1.5
# )

# # decision boundary
# plt.axhline(
#     y=50,
#     linestyle="--",
#     alpha=0.6
# )

# plt.xlabel(
#     "Angle in FaceNet PCA space"
# )

# plt.ylabel(
#     "Classifier probability (%)"
# )

# plt.title(
#     "Initial ArcFace Classifier Probability Before Self-Training"
# )

# plt.ylim(
#     0,
#     100
# )

# plt.xlim(
#     0,
#     360
# )

# plt.grid(
#     True,
#     alpha=0.3
# )

# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         ROOT_DIR,
#         "initial_arcface_probabilities.png"
#     ),
#     dpi=300
# )

# plt.close()

# plt.figure(
#     figsize=(12, 6)
# )

# plt.scatter(
#     df["angle_deg"],
#     df["prob_A"],
#     s=15
# )

# plt.plot(
#     df["angle_deg"],
#     df["prob_A"],
#     linewidth=1
# )

# plt.axhline(
#     y=0.5,
#     linestyle="--",
#     label="Decision boundary"
# )

# plt.xlabel(
#     "Angle in FaceNet PCA space"
# )

# plt.ylabel(
#     "P(A)"
# )

# plt.title(
#     "ArcFace Classifier P(A) Along FaceNet Rotation"
# )

# plt.xlim(
#     0,
#     360
# )

# plt.ylim(
#     0,
#     1
# )

# plt.grid(
#     True,
#     alpha=0.3
# )

# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         ROOT_DIR,
#         "arcface_probA_raw.png"
#     ),
#     dpi=300
# )

# plt.close()
# # ============================================================
# # SAVE RAW CLASSIFICATIONS
# # ============================================================

# df.to_csv(
#     "initial_arcface_predictions_perceptron.csv",
#     index=False
# )

# print(
#     "Saved graph:",
#     OUTPUT_PATH
# )

# print(
#     "Saved predictions:",
#     "initial_arcface_predictions_perceptron.csv"
# )

# # ============================================================
# # COMPUTE MEAN SOFTMAX PROBABILITY IN 20° WINDOWS
# # ============================================================

# window_size = 20

# results = []

# for step_angle in range(0, 360):

#     end = (step_angle + window_size) % 360

#     if step_angle < end:

#         window_data = df[
#             (df["angle_deg"] >= step_angle)
#             &
#             (df["angle_deg"] < end)
#         ]

#     else:

#         window_data = df[
#             (df["angle_deg"] >= step_angle)
#             |
#             (df["angle_deg"] < end)
#         ]

#     if len(window_data) > 0:

#         mean_prob_a = window_data[
#             "prob_A"
#         ].mean()

#         mean_prob_b = window_data[
#             "prob_B"
#         ].mean()

#     else:

#         mean_prob_a = np.nan
#         mean_prob_b = np.nan

#     center_angle = (
#         step_angle
#         + window_size / 2
#     ) % 360

#     results.append(
#         (
#             center_angle,
#             mean_prob_a,
#             mean_prob_b
#         )
#     )


# results_df = pd.DataFrame(
#     results,
#     columns=[
#         "angle",
#         "mean_prob_A",
#         "mean_prob_B"
#     ]
# )

# results_df = results_df.sort_values(
#     "angle"
# )


# # ============================================================
# # PLOT MEAN PROBABILITIES
# # ============================================================

# plt.figure(
#     figsize=(12, 6)
# )

# plt.plot(
#     results_df["angle"],
#     results_df["mean_prob_A"] * 100,
#     label="Mean P(A)",
#     linewidth=2
# )

# plt.plot(
#     results_df["angle"],
#     results_df["mean_prob_B"] * 100,
#     label="Mean P(B)",
#     linewidth=2
# )

# plt.axhline(
#     y=50,
#     linestyle="--",
#     alpha=0.6,
#     label="Decision boundary"
# )

# plt.xlabel(
#     "Angle in FaceNet PCA space"
# )

# plt.ylabel(
#     "Mean classifier probability (%)"
# )

# plt.title(
#     "ArcFace Mean Classification Probability in 20° Windows"
# )

# plt.xlim(
#     0,
#     360
# )

# plt.ylim(
#     0,
#     100
# )

# plt.grid(
#     True,
#     alpha=0.3
# )

# plt.legend()

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         ROOT_DIR,
#         "arcface_mean_probability_20deg.png"
#     ),
#     dpi=300
# )

# plt.close()


# # ============================================================
# # SAVE WINDOW RESULTS
# # ============================================================

# results_df.to_csv(
#     os.path.join(
#         ROOT_DIR,
#         "arcface_mean_probability_20deg.csv"
#     ),
#     index=False
# )