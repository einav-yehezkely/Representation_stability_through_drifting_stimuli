############################################################
# TEST TRAINED FACENET-EMBEDDING MLP ALONG PCA ROTATION PATH
#
# PCA is used ONLY to:
#   1. build the circular trajectory
#   2. select which image corresponds to each angle
#
# The classifier NEVER receives PCA coordinates.
#
# Model input:
#
# filename
#    ↓
# precomputed FaceNet 512D embedding
#    ↓
# MLP: 512 -> 64 -> ReLU -> 2
#    ↓
# P(A), P(B)
############################################################

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    SCRIPT_DIR
)


PCA_CSV = os.path.join(
    PROJECT_ROOT,
    "pca_top2_filtered_female.csv"
)


EMBEDDINGS_CSV = os.path.join(
    PROJECT_ROOT,
    "female_facenet_embeddings.csv"
)


# If the model file is in the same folder as this script,
# this is correct.
TRAINED_MODEL = os.path.join(
    PROJECT_ROOT,
    "model_ft_0_CE_FACENET.pth"
)


OUTPUT_PATH_CSV = (
    "facenet_pca_rotation_sequence.csv"
)

OUTPUT_RESULTS_CSV = (
    "facenet_initial_model_probabilities_pca_path.csv"
)

OUTPUT_GRAPH = (
    "facenet_initial_model_probability_pca_path.png"
)

OUTPUT_PATH_GRAPH = (
    "facenet_pca_selected_rotation_path.png"
)


TARGET_RADIUS = 0.45

ANGLE_STEP = 1.0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD FACENET EMBEDDINGS
#
# CSV format:
#
# column 0      = filename
# columns 1-512 = FaceNet embedding
#
# No header.
# ============================================================

print()
print("=" * 80)
print("LOADING PRECOMPUTED FACENET EMBEDDINGS")
print("=" * 80)


embeddings_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None,
)


if embeddings_df.shape[1] != 513:

    raise RuntimeError(
        f"Expected 513 columns in "
        f"{EMBEDDINGS_CSV} "
        f"(filename + 512 embedding dimensions), "
        f"but found {embeddings_df.shape[1]}"
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


    embedding_lookup[
        filename
    ] = embedding

    embedding_lookup[
        basename
    ] = embedding


print(
    "Number of FaceNet embeddings:",
    len(embeddings_df)
)

print(
    "Embedding dimension:",
    512
)

print(
    "Example filename:",
    embeddings_df.iloc[0, 0]
)

print(
    "Example embedding shape:",
    embedding_lookup[
        os.path.basename(
            str(
                embeddings_df.iloc[0, 0]
            )
        )
    ].shape
)

print("=" * 80)


# ============================================================
# MODEL
#
# MUST be identical to the model used during training.
# ============================================================

class FaceNetClassifier(nn.Module):

    def __init__(self):

        super().__init__()


        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2,
            ),
        )


    def forward(
        self,
        embeddings,
    ):

        return self.classifier(
            embeddings
        )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("=" * 80)
print("LOADING TRAINED MODEL")
print("=" * 80)

print(
    "Model path:",
    TRAINED_MODEL
)


if not os.path.exists(
    TRAINED_MODEL
):

    raise FileNotFoundError(
        f"Could not find trained model:\n"
        f"{TRAINED_MODEL}"
    )


model = FaceNetClassifier()


checkpoint = torch.load(
    TRAINED_MODEL,
    map_location=DEVICE,
)


# Support both:
#
# torch.save(model.state_dict(), ...)
#
# and:
#
# torch.save({
#     "model_state_dict": model.state_dict()
# }, ...)
#
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


model = model.to(
    DEVICE
)

model.eval()


print(
    "Model loaded successfully."
)

print(
    "Classifier: 512 -> 64 -> ReLU -> 2"
)

print("=" * 80)


# ============================================================
# LOAD PCA CSV
#
# PCA is ONLY used for constructing the path.
# ============================================================

print()
print("=" * 80)
print("LOADING PCA SPACE")
print("=" * 80)


pca_df = pd.read_csv(
    PCA_CSV,
    header=None,
    names=[
        "filename",
        "pc1",
        "pc2",
    ],
)


print(
    "Number of PCA images:",
    len(pca_df)
)


# ============================================================
# VERIFY THAT PCA IMAGES HAVE FACENET EMBEDDINGS
# ============================================================

missing_embeddings = []


for filename in pca_df[
    "filename"
]:

    basename = os.path.basename(
        str(filename).strip()
    )


    if basename not in embedding_lookup:

        missing_embeddings.append(
            basename
        )


print(
    "PCA images without FaceNet embedding:",
    len(missing_embeddings)
)


if missing_embeddings:

    print(
        "Examples:",
        missing_embeddings[:10]
    )


print("=" * 80)


# ============================================================
# CALCULATE PCA ANGLES / RADII
# ============================================================

pca_df["radius"] = np.sqrt(

    pca_df["pc1"] ** 2

    +

    pca_df["pc2"] ** 2
)


pca_df["angle"] = np.degrees(

    np.arctan2(

        pca_df["pc2"],

        pca_df["pc1"],
    )
)


pca_df["angle"] = (

    pca_df["angle"]

    +

    360

) % 360


# ============================================================
# BUILD IDEAL PCA ROTATION PATH
#
# One target every ANGLE_STEP degrees.
#
# For every target:
#
# target = (
#     r*cos(theta),
#     r*sin(theta)
# )
#
# Then choose the nearest UNUSED image.
#
# IMPORTANT:
# We only select images that also have a FaceNet embedding.
# ============================================================

target_angles = np.arange(
    0,
    360,
    ANGLE_STEP,
)


used_indices = set()

selected_rows = []


print()
print("=" * 80)
print("BUILDING PCA ROTATION PATH")
print("=" * 80)


pc1_values = pca_df[
    "pc1"
].to_numpy()


pc2_values = pca_df[
    "pc2"
].to_numpy()


# ------------------------------------------------------------
# Mark which PCA rows have a corresponding FaceNet embedding.
# ------------------------------------------------------------

has_embedding = np.array(
    [
        os.path.basename(
            str(filename).strip()
        ) in embedding_lookup

        for filename in pca_df[
            "filename"
        ]
    ],
    dtype=bool,
)


for step, target_angle in enumerate(
    target_angles
):

    theta = np.radians(
        target_angle
    )


    target_pc1 = (
        TARGET_RADIUS
        *
        np.cos(theta)
    )


    target_pc2 = (
        TARGET_RADIUS
        *
        np.sin(theta)
    )


    distances = np.sqrt(

        (
            pc1_values
            -
            target_pc1
        ) ** 2

        +

        (
            pc2_values
            -
            target_pc2
        ) ** 2
    )


    # --------------------------------------------------------
    # Never select a PCA point that has no FaceNet embedding.
    # --------------------------------------------------------

    distances[
        ~has_embedding
    ] = np.inf


    # --------------------------------------------------------
    # Do not reuse an image.
    # --------------------------------------------------------

    if used_indices:

        used_list = np.fromiter(
            used_indices,
            dtype=int,
        )


        distances[
            used_list
        ] = np.inf


    nearest_index = int(
        np.argmin(
            distances
        )
    )


    if not np.isfinite(
        distances[
            nearest_index
        ]
    ):

        raise RuntimeError(
            f"No valid unused image found "
            f"for target angle {target_angle}"
        )


    nearest_row = pca_df.iloc[
        nearest_index
    ]


    used_indices.add(
        nearest_index
    )


    selected_pc1 = float(
        nearest_row[
            "pc1"
        ]
    )


    selected_pc2 = float(
        nearest_row[
            "pc2"
        ]
    )


    selected_radius = float(
        nearest_row[
            "radius"
        ]
    )


    selected_angle = float(
        nearest_row[
            "angle"
        ]
    )


    distance_to_target = float(
        distances[
            nearest_index
        ]
    )


    filename = os.path.basename(
        str(
            nearest_row[
                "filename"
            ]
        ).strip()
    )


    selected_rows.append({

        "step":
            step,

        "target_angle":
            target_angle,

        "target_pc1":
            target_pc1,

        "target_pc2":
            target_pc2,

        "filename":
            filename,

        "pc1":
            selected_pc1,

        "pc2":
            selected_pc2,

        "actual_angle":
            selected_angle,

        "radius":
            selected_radius,

        "distance_to_target":
            distance_to_target,
    })


    print(
        f"{step:3d} | "
        f"target={target_angle:6.1f}° | "
        f"actual={selected_angle:7.2f}° | "
        f"distance={distance_to_target:.5f} | "
        f"{filename}"
    )


rotation_df = pd.DataFrame(
    selected_rows
)


rotation_df.to_csv(
    OUTPUT_PATH_CSV,
    index=False,
)


print()
print(
    "Saved PCA rotation sequence:",
    OUTPUT_PATH_CSV
)


# ============================================================
# VISUALIZE PCA PATH
# ============================================================

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    pca_df["pc1"],
    pca_df["pc2"],
    s=3,
    alpha=0.15,
    label="All FaceNet PCA images",
)


plt.scatter(
    rotation_df["pc1"],
    rotation_df["pc2"],
    s=15,
    label="Selected path",
)


plt.plot(
    rotation_df["pc1"],
    rotation_df["pc2"],
    linewidth=1,
)


circle_angles = np.linspace(
    0,
    2 * np.pi,
    500,
)


circle_x = (
    TARGET_RADIUS
    *
    np.cos(
        circle_angles
    )
)


circle_y = (
    TARGET_RADIUS
    *
    np.sin(
        circle_angles
    )
)


plt.plot(
    circle_x,
    circle_y,
    linestyle="--",
    linewidth=1,
    label="Ideal radius 0.45",
)


plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.title(
    "FaceNet PCA selected rotation path"
)

plt.axis(
    "equal"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_PATH_GRAPH,
    dpi=200,
)


plt.close()


print(
    "Saved path graph:",
    OUTPUT_PATH_GRAPH
)


# ============================================================
# LOAD PRECOMPUTED EMBEDDING
#
# IMPORTANT:
# No image is opened here.
# ============================================================

def load_embedding(
    filename,
):

    filename = str(
        filename
    ).strip()


    basename = os.path.basename(
        filename
    )


    if basename not in embedding_lookup:

        raise KeyError(
            f"No FaceNet embedding found for: "
            f"{filename}"
        )


    embedding = embedding_lookup[
        basename
    ]


    embedding = torch.tensor(
        embedding,
        dtype=torch.float32,
    )


    # Add batch dimension:
    #
    # [512] -> [1, 512]
    embedding = embedding.unsqueeze(
        0
    )


    return embedding.to(
        DEVICE
    )


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(
    embedding,
):

    model.eval()


    logits = model(
        embedding
    )


    probabilities = torch.softmax(
        logits,
        dim=1,
    )


    pA = probabilities[
        0,
        0
    ].item()


    pB = probabilities[
        0,
        1
    ].item()


    prediction = int(

        torch.argmax(
            probabilities,
            dim=1,
        ).item()
    )


    return (
        prediction,
        pA,
        pB,
    )


# ============================================================
# TEST INITIAL CLASSIFIER ON PCA PATH
#
# IMPORTANT:
#
# PCA selects the image.
#
# But the classifier receives ONLY:
#
#     512D FaceNet embedding
#
# There is NO training here.
# ============================================================

results = []


print()
print("=" * 80)
print("TESTING FACENET-EMBEDDING MODEL ON PCA ROTATION PATH")
print("=" * 80)
print()


for i, row in rotation_df.iterrows():

    filename = str(
        row[
            "filename"
        ]
    )


    target_angle = float(
        row[
            "target_angle"
        ]
    )


    actual_angle = float(
        row[
            "actual_angle"
        ]
    )


    # --------------------------------------------------------
    # Model sees ONLY FaceNet representation.
    # --------------------------------------------------------

    embedding = load_embedding(
        filename
    )


    prediction, pA, pB = predict(
        embedding
    )


    predicted_name = (
        "A"
        if prediction == 0
        else "B"
    )


    print(
        f"{i:3d} | "
        f"target={target_angle:6.1f}° | "
        f"actual={actual_angle:7.2f}° | "
        f"{filename:15s} | "
        f"pred={predicted_name} | "
        f"P(A)={pA:.4f} | "
        f"P(B)={pB:.4f}"
    )


    results.append({

        "step":
            i,

        "target_angle":
            target_angle,

        "actual_angle":
            actual_angle,

        "filename":
            filename,

        "pc1":
            row[
                "pc1"
            ],

        "pc2":
            row[
                "pc2"
            ],

        "radius":
            row[
                "radius"
            ],

        "distance_to_target":
            row[
                "distance_to_target"
            ],

        "prediction":
            predicted_name,

        "pA":
            pA,

        "pB":
            pB,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_RESULTS_CSV,
    index=False,
)


print()
print(
    "Saved model results:",
    OUTPUT_RESULTS_CSV
)


# ============================================================
# PLOT PROBABILITIES
# ============================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    results_df[
        "target_angle"
    ],
    results_df[
        "pA"
    ],
    label="P(A)",
    linewidth=2,
)


plt.plot(
    results_df[
        "target_angle"
    ],
    results_df[
        "pB"
    ],
    label="P(B)",
    linewidth=2,
)


plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="decision boundary (0.5)",
)


plt.xlabel(
    "Target PCA angle (degrees)"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "FaceNet 512D MLP prediction along PCA rotation path\n"
    "PCA selects images; classifier sees only FaceNet embeddings"
)

plt.xlim(
    0,
    359
)

plt.ylim(
    -0.02,
    1.02
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_GRAPH,
    dpi=200,
)


plt.close()


print(
    "Saved probability graph:",
    OUTPUT_GRAPH
)


# ============================================================
# SUMMARY
# ============================================================

num_A = (
    results_df[
        "prediction"
    ]
    ==
    "A"
).sum()


num_B = (
    results_df[
        "prediction"
    ]
    ==
    "B"
).sum()


predictions = results_df[
    "prediction"
].to_numpy()


num_switches = np.sum(
    predictions[1:]
    !=
    predictions[:-1]
)


# Circular switch:
# compare final point with first point.
circular_switch = int(
    predictions[-1]
    !=
    predictions[0]
)


num_switches_circular = (
    num_switches
    +
    circular_switch
)


mean_target_distance = (
    results_df[
        "distance_to_target"
    ].mean()
)


max_target_distance = (
    results_df[
        "distance_to_target"
    ].max()
)


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)


print(
    f"Number of selected images: "
    f"{len(results_df)}"
)


print(
    f"Unique images: "
    f"{results_df['filename'].nunique()}"
)


print(
    f"Predicted A: "
    f"{num_A}/{len(results_df)} "
    f"({100 * num_A / len(results_df):.1f}%)"
)


print(
    f"Predicted B: "
    f"{num_B}/{len(results_df)} "
    f"({100 * num_B / len(results_df):.1f}%)"
)


print(
    f"A/B switches along path: "
    f"{num_switches}"
)


print(
    f"A/B switches around full circle: "
    f"{num_switches_circular}"
)


print(
    f"Mean distance to ideal PCA point: "
    f"{mean_target_distance:.6f}"
)


print(
    f"Maximum distance to ideal PCA point: "
    f"{max_target_distance:.6f}"
)


print(
    f"Minimum P(A): "
    f"{results_df['pA'].min():.6f}"
)


print(
    f"Maximum P(A): "
    f"{results_df['pA'].max():.6f}"
)


print("=" * 80)