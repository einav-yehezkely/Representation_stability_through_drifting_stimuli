import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# CONFIG
# ============================================================

UMAP_CSV = "shufflenet_umap.csv"

SEQUENCE_CSV = os.path.join(
    "shufflenet_smooth_rotation_test",
    "smooth_rotation_sequence.csv",
)

MODEL_PATH = "model_ft_0_CE_UMAP_last_layer.pth"

EMBEDDINGS_CSV = os.path.join(
    "shufflenetSpace",
    "shufflenet_embeddings.csv",
)

OUTPUT_DIR = "output_two_opposite_256space"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# False = supervised:
# current image = A, opposite-in-256D image = B
#
# True = self-learning:
# each image's own prediction becomes its pseudo-label
UNSUPERVISED = False

LEARNING_RATE = 0.1
L2_LAMBDA = 0.1

SAVE_FRAME_EVERY = 5

# Keep the trained bias fixed during online learning.
FREEZE_BIAS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD UMAP + SMOOTH SEQUENCE
# ============================================================

def load_umap_and_sequence():
    umap_df = pd.read_csv(UMAP_CSV, header=0)
    umap_df.columns = ["filename", "x", "y"]

    seq_df = pd.read_csv(SEQUENCE_CSV)

    required = {"filename", "angle"}
    missing = required - set(seq_df.columns)

    if missing:
        raise ValueError(
            f"{SEQUENCE_CSV} must contain {required}. "
            f"Found columns: {list(seq_df.columns)}"
        )

    seq_df = seq_df.merge(
        umap_df[["filename", "x", "y"]],
        on="filename",
        how="left",
    )

    bad = seq_df[seq_df["x"].isna() | seq_df["y"].isna()]
    if len(bad) > 0:
        raise ValueError(
            "Some smooth-sequence images were not found in the UMAP CSV:\n"
            + "\n".join(bad["filename"].head(20).tolist())
        )

    umap_centroid = umap_df[["x", "y"]].to_numpy().mean(axis=0)

    return umap_df, seq_df, umap_centroid


# ============================================================
# LOAD SAVED 1024-D SHUFFLENET EMBEDDINGS
# ============================================================

def load_1024_embeddings():
    df = pd.read_csv(EMBEDDINGS_CSV)

    if "filename" not in df.columns:
        raise ValueError(
            f"{EMBEDDINGS_CSV} must contain a 'filename' column."
        )

    filenames = df["filename"].tolist()

    embeddings = (
        df.drop(columns=["filename"])
        .values
        .astype(np.float32)
    )

    print(
        f"Loaded {len(filenames)} cached ShuffleNet embeddings "
        f"| dim={embeddings.shape[1]}"
    )

    return filenames, embeddings


# ============================================================
# LOAD TRAINED 1024 -> 256 -> 2 MODEL
# ============================================================

def load_trained_model():
    model = models.shufflenet_v2_x0_5(weights=None)

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()

    print("Loaded trained 1024 -> 256 -> 2 classifier.")

    return model


# ============================================================
# CREATE TRAINED 256-D REPRESENTATIONS
# ============================================================

@torch.no_grad()
def convert_1024_to_trained_256(
    model,
    embeddings_1024,
    batch_size=4096,
):
    """
    Uses the already-trained:
        Linear(1024 -> 256) + ReLU

    No image loading is needed.
    """

    linear_1024_to_256 = model.fc[1]
    relu = model.fc[2]

    outputs = []

    for start in range(
        0,
        len(embeddings_1024),
        batch_size,
    ):
        batch_np = embeddings_1024[
            start:start + batch_size
        ]

        batch = torch.tensor(
            batch_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        h = relu(
            linear_1024_to_256(batch)
        )

        outputs.append(
            h.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    features_256 = np.concatenate(
        outputs,
        axis=0,
    )

    print(
        "Converted cached embeddings to trained "
        f"256-D representation | shape={features_256.shape}"
    )

    return features_256


# ============================================================
# BINARY VERSION OF TRAINED FINAL LAYER
# ============================================================

class TrainedBinaryHead(nn.Module):
    """
    Original final layer:
        z_A = W_A h + b_A
        z_B = W_B h + b_B

    Equivalent binary logit:
        z = z_B - z_A
    """

    def __init__(self, trained_final_linear):
        super().__init__()

        with torch.no_grad():
            initial_w = (
                trained_final_linear.weight[1]
                - trained_final_linear.weight[0]
            ).detach().clone()

            if trained_final_linear.bias is None:
                initial_b = torch.tensor(
                    0.0,
                    device=initial_w.device,
                )
            else:
                initial_b = (
                    trained_final_linear.bias[1]
                    - trained_final_linear.bias[0]
                ).detach().clone()

        self.weight = nn.Parameter(initial_w)
        self.bias = nn.Parameter(initial_b)

    def forward(self, h):
        return h @ self.weight + self.bias


def build_trained_binary_head(model):
    head = TrainedBinaryHead(
        model.fc[4]
    ).to(DEVICE)

    if FREEZE_BIAS:
        head.bias.requires_grad_(False)

    return head


# ============================================================
# FIND OPPOSITE IMAGE IN THE 256-D CLASSIFIER SPACE
# ============================================================

def build_opposite_lookup_256(
    seq_df,
    filenames,
    features_256,
    feature_index,
):
    """
    For each trajectory image A:

    1. Center the trained 256-D representation space.
    2. L2-normalize each centered feature.
    3. Choose B as the real image with MINIMUM cosine similarity
       to A in this centered 256-D space.

    So B is as close as possible to the opposite direction of A
    in the SAME representation space where SGD is performed.

    UMAP is NOT used for choosing B.
    """

    print()
    print(
        "Building opposite-image lookup in trained 256-D space..."
    )

    # Center the representation space.
    mean_256 = features_256.mean(
        axis=0,
        keepdims=True,
    )

    centered = (
        features_256
        - mean_256
    ).astype(np.float32)

    # Normalize all candidate directions.
    norms = np.linalg.norm(
        centered,
        axis=1,
        keepdims=True,
    )

    norms[norms < 1e-12] = 1.0

    normalized = (
        centered / norms
    ).astype(np.float32)

    # Build all trajectory queries at once.
    valid_rows = []
    query_indices = []

    for _, row in seq_df.iterrows():
        fn = row["filename"]

        if fn in feature_index:
            valid_rows.append(fn)
            query_indices.append(
                feature_index[fn]
            )

    query_indices = np.asarray(
        query_indices,
        dtype=np.int64,
    )

    queries = normalized[
        query_indices
    ]

    # Matrix of cosine similarities:
    # [num_path_images, num_all_images]
    #
    # Most opposite = smallest cosine similarity.
    print(
        f"Computing cosine similarities: "
        f"{queries.shape[0]} trajectory images x "
        f"{normalized.shape[0]} candidate images..."
    )

    similarities = (
        queries @ normalized.T
    )

    # Never select the same image as its own opposite.
    similarities[
        np.arange(len(query_indices)),
        query_indices,
    ] = np.inf

    opposite_indices = np.argmin(
        similarities,
        axis=1,
    )

    opposite_scores = similarities[
        np.arange(len(opposite_indices)),
        opposite_indices,
    ]

    lookup = {}
    score_lookup = {}

    filenames_arr = np.asarray(
        filenames,
        dtype=object,
    )

    for fn_A, idx_B, score in zip(
        valid_rows,
        opposite_indices,
        opposite_scores,
    ):
        lookup[fn_A] = filenames_arr[
            int(idx_B)
        ]

        score_lookup[fn_A] = float(
            score
        )

    scores = np.asarray(
        list(score_lookup.values()),
        dtype=np.float32,
    )

    print(
        "Opposite lookup completed."
    )

    print(
        f"Cosine similarity A vs chosen B: "
        f"mean={scores.mean():.4f}, "
        f"min={scores.min():.4f}, "
        f"max={scores.max():.4f}"
    )

    return lookup, score_lookup


# ============================================================
# ONE JOINT SGD STEP ON A + B
# ============================================================

def one_pair_sgd_step(
    head,
    h_A,
    h_B,
    label_A,
    label_B,
):
    """
    One joint backward pass and one SGD update on two images.
    """

    head.zero_grad(
        set_to_none=True
    )

    h_pair = torch.cat(
        [h_A, h_B],
        dim=0,
    )

    targets = torch.tensor(
        [
            float(label_A),
            float(label_B),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    logits = head(h_pair)

    classification_loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="mean",
        )
    )

    classification_loss.backward()

    with torch.no_grad():
        grad_classification = (
            head.weight.grad
            .detach()
            .clone()
        )

        grad_l2 = (
            L2_LAMBDA
            * head.weight
            .detach()
            .clone()
        )

        grad_total = (
            grad_classification
            + grad_l2
        )

        head.weight -= (
            LEARNING_RATE
            * grad_total
        )

        if (
            head.bias.requires_grad
            and head.bias.grad is not None
        ):
            head.bias -= (
                LEARNING_RATE
                * head.bias.grad
            )

        class_norm = float(
            grad_classification
            .norm()
            .cpu()
        )

        l2_norm = float(
            grad_l2
            .norm()
            .cpu()
        )

        ratio = (
            class_norm
            / (l2_norm + 1e-12)
        )

    return {
        "loss": float(
            classification_loss
            .detach()
            .cpu()
        ),

        "classification_grad_norm": (
            class_norm
        ),

        "l2_grad_norm": (
            l2_norm
        ),

        "class_to_l2_ratio": (
            ratio
        ),

        "total_grad_norm": float(
            grad_total
            .norm()
            .cpu()
        ),
    }


# ============================================================
# CLASSIFY ALL CACHED 256-D FEATURES
# ============================================================

@torch.no_grad()
def predict_all(
    head,
    features_256,
    batch_size=20000,
):
    all_preds = []
    all_probs = []

    for start in range(
        0,
        len(features_256),
        batch_size,
    ):
        batch_np = features_256[
            start:start + batch_size
        ]

        h = torch.tensor(
            batch_np,
            dtype=torch.float32,
            device=DEVICE,
        )

        logits = head(h)

        probs_B = torch.sigmoid(
            logits
        )

        preds = (
            probs_B >= 0.5
        ).long()

        all_preds.append(
            preds.cpu().numpy()
        )

        all_probs.append(
            probs_B.cpu().numpy()
        )

    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


# ============================================================
# EMPIRICAL UMAP CLASSIFIER ANGLE
# ============================================================

def estimate_empirical_model_angle(
    vis_df,
    preds,
    centroid,
):
    """
    UMAP is used only for this visualization/angle estimate.
    """

    df = vis_df.copy()
    df["pred"] = preds

    df["angle_deg"] = (
        np.degrees(
            np.arctan2(
                df["y"] - centroid[1],
                df["x"] - centroid[0],
            )
        )
        % 360
    )

    df = df.sort_values(
        "angle_deg"
    )

    if len(df) < 2:
        return np.nan

    angles = df[
        "angle_deg"
    ].to_numpy()

    labels = df[
        "pred"
    ].to_numpy()

    changes = np.where(
        labels[:-1]
        != labels[1:]
    )[0]

    boundaries = []

    for idx in changes:
        boundaries.append(
            (
                angles[idx]
                + angles[idx + 1]
            )
            / 2.0
        )

    if labels[-1] != labels[0]:
        boundaries.append(
            (
                (
                    angles[-1]
                    + angles[0]
                    + 360
                )
                / 2.0
            )
            % 360
        )

    if len(boundaries) == 0:
        return np.nan

    boundaries = np.asarray(
        boundaries
    )

    boundary_180 = (
        boundaries % 180
    )

    doubled = np.deg2rad(
        boundary_180 * 2
    )

    mean_boundary = (
        np.degrees(
            np.arctan2(
                np.mean(
                    np.sin(doubled)
                ),
                np.mean(
                    np.cos(doubled)
                ),
            )
        )
        / 2.0
    ) % 180

    return (
        mean_boundary + 90
    ) % 180


# ============================================================
# SAVE UMAP FRAME
# ============================================================

def save_boundary_frame(
    vis_df,
    vis_features,
    seq_df,
    centroid,
    iteration,
    head,
    record,
):
    preds, probs = predict_all(
        head,
        vis_features,
    )

    df = vis_df.copy()
    df["pred"] = preds
    df["prob_B"] = probs

    df_A = df[
        df["pred"] == 0
    ]

    df_B = df[
        df["pred"] == 1
    ]

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    ax.scatter(
        df_A["x"],
        df_A["y"],
        s=4,
        alpha=0.30,
        color="blue",
        label=(
            f"Predicted A "
            f"({len(df_A):,})"
        ),
    )

    ax.scatter(
        df_B["x"],
        df_B["y"],
        s=4,
        alpha=0.30,
        color="red",
        label=(
            f"Predicted B "
            f"({len(df_B):,})"
        ),
    )

    path = seq_df.iloc[
        :iteration + 1
    ]

    ax.plot(
        path["x"],
        path["y"],
        linewidth=2,
        color="black",
        label="Smooth UMAP trajectory",
    )

    # Current A image
    ax.scatter(
        [record["A_x"]],
        [record["A_y"]],
        marker="*",
        s=260,
        color="gold",
        edgecolors="black",
        label="Current A image",
        zorder=10,
    )

    # B chosen by opposite direction in 256-D
    ax.scatter(
        [record["B_x"]],
        [record["B_y"]],
        marker="*",
        s=260,
        color="lime",
        edgecolors="black",
        label="256-D opposite B image",
        zorder=10,
    )

    ax.scatter(
        [centroid[0]],
        [centroid[1]],
        marker="+",
        s=140,
        color="black",
        label="UMAP centroid",
        zorder=10,
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_title(
        f"Iteration {iteration}\n"
        f"A P(B)={record['prob_B_A_before']:.3f} | "
        f"B P(B)={record['prob_B_B_before']:.3f} | "
        f"cos256(A,B)={record['opposite_cosine_256']:.3f} | "
        f"global A={record['percent_A_all']:.1f}% "
        f"B={record['percent_B_all']:.1f}%"
    )

    # Fixed position: much faster than loc='best' with 118k points.
    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"boundary_{iteration:04d}.png",
        ),
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# SUMMARY PLOTS
# ============================================================

def save_summary_plots(df):
    # Angle tracking
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["example_angle"] % 180,
        label="examples",
    )

    ax.plot(
        df["iteration"],
        df["model_angle"],
        label="classifier direction",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("angle [degrees]")
    ax.set_ylim(0, 180)
    ax.set_yticks(
        np.arange(0, 181, 20)
    )

    ax.set_title(
        "Opposite examples chosen in 256-D space"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "angle_tracking_graph.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # A/B probabilities
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["prob_B_A_before"],
        label="A image: P(B)",
    )

    ax.plot(
        df["iteration"],
        df["prob_B_B_before"],
        label="256-D opposite B image: P(B)",
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color="black",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("P(B)")
    ax.set_ylim(0, 1)

    ax.set_title(
        "Predictions before each pair update"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "pair_probabilities.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # Global class balance
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["percent_A_all"],
        label="% predicted A",
    )

    ax.plot(
        df["iteration"],
        df["percent_B_all"],
        label="% predicted B",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("% of all images")
    ax.set_ylim(0, 100)

    ax.set_title(
        "Global classifier balance"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "global_class_balance.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # Weight norm
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["weight_norm"],
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("||w||")

    ax.set_title(
        "256-D classifier weight norm"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "weight_norm.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # Gradient comparison
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["classification_grad_norm"],
        label="pair classification gradient",
    )

    ax.plot(
        df["iteration"],
        df["l2_grad_norm"],
        label="L2 gradient",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("gradient norm")

    ax.set_title(
        "Pair gradient vs L2 gradient"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gradient_norms.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # Gradient ratio
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["class_to_l2_ratio"],
    )

    ax.axhline(
        1.0,
        linestyle="--",
        color="black",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel(
        "||classification grad|| / ||L2 grad||"
    )

    ax.set_title(
        "Classification-vs-L2 gradient ratio"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "gradient_ratio.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # How opposite the selected pair really is in 256-D
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["opposite_cosine_256"],
    )

    ax.axhline(
        0.0,
        linestyle="--",
        color="black",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel(
        "cosine similarity in centered 256-D"
    )

    ax.set_title(
        "A vs selected opposite B in 256-D"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "opposite_cosine_256.png",
        ),
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def run():
    print(
        "Device:",
        DEVICE,
    )

    # UMAP + trajectory
    umap_df, seq_df, umap_centroid = (
        load_umap_and_sequence()
    )

    print(
        "Smooth trajectory steps:",
        len(seq_df),
    )

    # Cached 1024-D embeddings
    filenames_1024, embeddings_1024 = (
        load_1024_embeddings()
    )

    # Trained model
    trained_model = (
        load_trained_model()
    )

    # Exact trained 256-D features
    features_256 = (
        convert_1024_to_trained_256(
            trained_model,
            embeddings_1024,
        )
    )

    # Trained final binary classifier
    head = (
        build_trained_binary_head(
            trained_model
        )
    )

    print(
        "Initial trained binary head "
        f"| ||w||="
        f"{head.weight.detach().norm().item():.6f}"
    )

    # filename -> feature row
    feature_index = {
        filename: i
        for i, filename
        in enumerate(
            filenames_1024
        )
    }

    # UMAP images with valid trained features
    vis_df = (
        umap_df[
            umap_df["filename"].isin(
                feature_index
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    vis_features = np.stack(
        [
            features_256[
                feature_index[filename]
            ]
            for filename
            in vis_df["filename"]
        ]
    )

    print(
        f"Using {len(vis_df):,} "
        "trained 256-D features."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Choose B in 256-D, NOT in UMAP.
    # --------------------------------------------------------
    opposite_lookup, opposite_score_lookup = (
        build_opposite_lookup_256(
            seq_df=seq_df,
            filenames=filenames_1024,
            features_256=features_256,
            feature_index=feature_index,
        )
    )

    # UMAP positions only for plotting the chosen images.
    vis_point_map = {
        row["filename"]: (
            float(row["x"]),
            float(row["y"]),
        )
        for _, row
        in vis_df.iterrows()
    }

    # Initial classifier state
    initial_preds, _ = (
        predict_all(
            head,
            vis_features,
        )
    )

    print()
    print(
        "Initial classifier: "
        f"A={(initial_preds == 0).mean()*100:.2f}% "
        f"B={(initial_preds == 1).mean()*100:.2f}%"
    )

    print()
    print(
        "Starting TWO-IMAGE experiment:"
    )

    if UNSUPERVISED:
        print(
            "A image + 256-D opposite image "
            "-> own predictions "
            "-> ONE joint SGD+L2 step"
        )
    else:
        print(
            "current image=A + "
            "256-D opposite image=B "
            "-> ONE joint SGD+L2 step"
        )

    print()

    history = []

    for i, row in seq_df.iterrows():
        filename_A = row["filename"]

        if filename_A not in feature_index:
            print(
                f"Skipping {filename_A}: "
                "missing 256-D feature."
            )
            continue

        if filename_A not in opposite_lookup:
            print(
                f"Skipping {filename_A}: "
                "no opposite image."
            )
            continue

        filename_B = opposite_lookup[
            filename_A
        ]

        # ------------------------
        # A and B in trained 256-D
        # ------------------------
        h_A = torch.tensor(
            features_256[
                feature_index[
                    filename_A
                ]
            ],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        h_B = torch.tensor(
            features_256[
                feature_index[
                    filename_B
                ]
            ],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        # Predictions before update
        with torch.no_grad():
            prob_B_A_before = float(
                torch.sigmoid(
                    head(h_A).squeeze()
                ).cpu()
            )

            prob_B_B_before = float(
                torch.sigmoid(
                    head(h_B).squeeze()
                ).cpu()
            )

        # Labels
        if UNSUPERVISED:
            label_A = int(
                prob_B_A_before >= 0.5
            )

            label_B = int(
                prob_B_B_before >= 0.5
            )

        else:
            label_A = 0
            label_B = 1

        # One joint update
        step_info = one_pair_sgd_step(
            head=head,
            h_A=h_A,
            h_B=h_B,
            label_A=label_A,
            label_B=label_B,
        )

        # Global classifier state after update
        all_preds, _ = (
            predict_all(
                head,
                vis_features,
            )
        )

        percent_A = float(
            (all_preds == 0).mean()
            * 100
        )

        percent_B = float(
            (all_preds == 1).mean()
            * 100
        )

        model_angle = (
            estimate_empirical_model_angle(
                vis_df,
                all_preds,
                umap_centroid,
            )
        )

        if filename_B in vis_point_map:
            B_x, B_y = (
                vis_point_map[
                    filename_B
                ]
            )
        else:
            B_x, B_y = (
                np.nan,
                np.nan,
            )

        rec = {
            "iteration": i,

            "filename_A": filename_A,
            "filename_B": filename_B,

            "example_angle": float(
                row["angle"]
            ),

            "A_x": float(
                row["x"]
            ),

            "A_y": float(
                row["y"]
            ),

            "B_x": B_x,
            "B_y": B_y,

            "opposite_cosine_256": float(
                opposite_score_lookup[
                    filename_A
                ]
            ),

            "prob_B_A_before": (
                prob_B_A_before
            ),

            "prob_B_B_before": (
                prob_B_B_before
            ),

            "label_A": label_A,
            "label_B": label_B,

            "model_angle": (
                model_angle
            ),

            "percent_A_all": (
                percent_A
            ),

            "percent_B_all": (
                percent_B
            ),

            "weight_norm": float(
                head.weight
                .detach()
                .norm()
                .cpu()
            ),

            **step_info,
        }

        history.append(rec)

        angle_text = (
            f"{model_angle:7.2f}°"
            if not np.isnan(model_angle)
            else "   NaN "
        )

        print(
            f"{i:4d} | "
            f"A={filename_A:12s} "
            f"P(B)={prob_B_A_before:.3f} | "
            f"B={filename_B:12s} "
            f"P(B)={prob_B_B_before:.3f} | "
            f"cos256={rec['opposite_cosine_256']:.3f} | "
            f"example={float(row['angle']):7.2f}° | "
            f"model={angle_text} | "
            f"global A={percent_A:6.2f}% "
            f"B={percent_B:6.2f}% | "
            f"ratio={rec['class_to_l2_ratio']:.3f}"
        )

        if (
            i == 0
            or i % SAVE_FRAME_EVERY == 0
            or i == len(seq_df) - 1
        ):
            save_boundary_frame(
                vis_df=vis_df,
                vis_features=vis_features,
                seq_df=seq_df,
                centroid=umap_centroid,
                iteration=i,
                head=head,
                record=rec,
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "online_learning_history.csv",
        ),
        index=False,
    )

    save_summary_plots(
        history_df
    )

    torch.save(
        {
            "weight": (
                head.weight
                .detach()
                .cpu()
            ),

            "bias": (
                head.bias
                .detach()
                .cpu()
            ),
        },
        os.path.join(
            OUTPUT_DIR,
            "final_binary_head_256.pth",
        ),
    )

    print()
    print("Finished.")
    print(
        "Output directory:",
        OUTPUT_DIR,
    )

    print()
    print("Main outputs:")
    print("  boundary_XXXX.png")
    print("  angle_tracking_graph.png")
    print("  pair_probabilities.png")
    print("  global_class_balance.png")
    print("  weight_norm.png")
    print("  gradient_norms.png")
    print("  gradient_ratio.png")
    print("  opposite_cosine_256.png")
    print("  online_learning_history.csv")


if __name__ == "__main__":
    run()