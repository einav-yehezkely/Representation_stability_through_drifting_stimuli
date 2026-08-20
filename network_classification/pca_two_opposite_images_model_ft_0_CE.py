import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
IMAGE_DIR = "female_faces"
MODEL_PATH = "model_ft_0_CE.pth"

OUTPUT_DIR = "output_pca_two_opposite_256_trained"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache created from the FULLY FINE-TUNED model.
# Important: this is NOT the old ImageNet ShuffleNet embedding cache.
FEATURE_CACHE = os.path.join(
    OUTPUT_DIR,
    "pca_model_ft_0_CE_256_features.npz",
)

# Supervised sanity experiment by default:
# current PCA-side image = A, opposite PCA-side image = B
UNSUPERVISED = False

# Online update
LEARNING_RATE = 0.1
L2_LAMBDA = 0.1

# PCA trajectory
NUM_STEPS = 360
ROTATION_RANGE_DEG = 360.0
START_ANGLE_DEG = 0.0
TARGET_RADIUS = 0.45

# One real image per side per step.
# Avoid reusing images across the full trajectory.
UNIQUE_IMAGES = True

# Visualization
SAVE_FRAME_EVERY = 5

# Feature extraction
FEATURE_BATCH_SIZE = 128
NUM_WORKERS = 0  # Windows-safe

# Keep the trained bias fixed during online updates.
FREEZE_BIAS = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA
# ============================================================

def load_pca():
    """
    pca_top2_filtered_female.csv format:
        filename, x, y
    with no header.
    """
    df = pd.read_csv(PCA_CSV, header=None)
    df.columns = ["filename", "x", "y"]

    df["filename"] = df["filename"].astype(str)

    df["angle_deg"] = (
        np.degrees(
            np.arctan2(
                df["y"].to_numpy(),
                df["x"].to_numpy(),
            )
        )
        % 360
    )

    df["radius"] = np.sqrt(
        df["x"] ** 2
        + df["y"] ** 2
    )

    return df


def circular_angle_distance_deg(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def find_initial_base_point(pca_df):
    """
    Same idea as the old PCA code:
    pick a real point near angle 0 and radius ~0.45.
    """
    angle_error = circular_angle_distance_deg(
        pca_df["angle_deg"].to_numpy(),
        START_ANGLE_DEG,
    )

    radius_error = np.abs(
        pca_df["radius"].to_numpy()
        - TARGET_RADIUS
    )

    score = (
        angle_error
        + 100.0 * radius_error
    )

    idx = int(np.argmin(score))

    return pca_df.loc[
        idx,
        ["x", "y"],
    ].to_numpy(dtype=np.float32)


def rotate_vector(v, angle_deg):
    a = np.deg2rad(angle_deg)

    R = np.array(
        [
            [np.cos(a), -np.sin(a)],
            [np.sin(a),  np.cos(a)],
        ],
        dtype=np.float32,
    )

    return R @ v


def generate_opposite_pca_sequence(pca_df):
    """
    Build a sequence:
        A_t near rotated base point
        B_t near -rotated base point

    Thus A and B are defined as opposite positions in the PCA plane.
    """

    names = pca_df["filename"].to_numpy()
    points = pca_df[
        ["x", "y"]
    ].to_numpy(dtype=np.float32)

    base_point = find_initial_base_point(
        pca_df
    )

    used = set()
    rows = []

    for step in range(NUM_STEPS):
        rotation = (
            START_ANGLE_DEG
            + ROTATION_RANGE_DEG
            * step
            / NUM_STEPS
        )

        target_A = rotate_vector(
            base_point,
            rotation,
        )

        target_B = -target_A

        dA = np.linalg.norm(
            points - target_A,
            axis=1,
        )

        dB = np.linalg.norm(
            points - target_B,
            axis=1,
        )

        if UNIQUE_IMAGES:
            if used:
                used_idx = np.fromiter(
                    used,
                    dtype=np.int64,
                )
                dA[used_idx] = np.inf
                dB[used_idx] = np.inf

        idx_A = int(np.argmin(dA))

        if UNIQUE_IMAGES:
            used.add(idx_A)
            dB[idx_A] = np.inf

        idx_B = int(np.argmin(dB))

        if UNIQUE_IMAGES:
            used.add(idx_B)

        actual_A = points[idx_A]
        actual_B = points[idx_B]

        angle_A = (
            np.degrees(
                np.arctan2(
                    actual_A[1],
                    actual_A[0],
                )
            )
            % 360
        )

        angle_B = (
            np.degrees(
                np.arctan2(
                    actual_B[1],
                    actual_B[0],
                )
            )
            % 360
        )

        rows.append(
            {
                "iteration": step,

                "target_angle_A": (
                    np.degrees(
                        np.arctan2(
                            target_A[1],
                            target_A[0],
                        )
                    )
                    % 360
                ),

                "filename_A": names[idx_A],
                "A_x": float(actual_A[0]),
                "A_y": float(actual_A[1]),
                "actual_angle_A": float(angle_A),

                "filename_B": names[idx_B],
                "B_x": float(actual_B[0]),
                "B_y": float(actual_B[1]),
                "actual_angle_B": float(angle_B),

                "distance_A_to_target": float(
                    dA[idx_A]
                ),

                "distance_B_to_target": float(
                    dB[idx_B]
                ),
            }
        )

    seq_df = pd.DataFrame(rows)

    seq_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pca_opposite_sequence.csv",
        ),
        index=False,
    )

    print(
        f"Generated {len(seq_df)} PCA pair steps."
    )

    return seq_df


# ============================================================
# MODEL — EXACT ARCHITECTURE OF model_ft_0_CE.pth
# ============================================================

def load_trained_model():
    """
    Architecture from the original training code:
        ShuffleNetV2 x0.5
        Dropout(0.5)
        Linear(... -> 256)
        ReLU
        Dropout(0.3)
        Linear(256 -> 2)

    model_ft_0_CE.pth was trained with full fine-tuning,
    so we must extract features using THIS checkpoint.
    """

    model = models.shufflenet_v2_x0_5(
        weights=None
    )

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(
            num_ftrs,
            256,
        ),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(
            256,
            2,
        ),
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state)

    model = model.to(DEVICE)
    model.eval()

    print(
        "Loaded model_ft_0_CE.pth."
    )

    return model


# ============================================================
# EXTRACT EXACT TRAINED 256-D FEATURES
# ============================================================

data_transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


class PCAImageDataset(Dataset):
    def __init__(
        self,
        filenames,
    ):
        self.filenames = list(
            filenames
        )

    def __len__(self):
        return len(
            self.filenames
        )

    def __getitem__(
        self,
        idx,
    ):
        filename = self.filenames[
            idx
        ]

        path = os.path.join(
            IMAGE_DIR,
            filename,
        )

        image = Image.open(
            path
        ).convert("RGB")

        image = data_transform(
            image
        )

        return image, filename


class TrainedPenultimate256(nn.Module):
    """
    Forward through the fully fine-tuned ShuffleNet and return
    the 256-D vector immediately before the last Linear(256,2).
    """

    def __init__(
        self,
        model,
    ):
        super().__init__()

        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.conv5 = model.conv5

        # In eval mode Dropout is identity.
        self.fc0 = model.fc[0]
        self.fc1 = model.fc[1]
        self.fc2 = model.fc[2]
        self.fc3 = model.fc[3]

    def forward(
        self,
        x,
    ):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)

        x = x.mean(
            [2, 3]
        )

        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


@torch.no_grad()
def extract_or_load_256_features(
    trained_model,
    pca_df,
):
    """
    First run:
      load PCA images -> trained model -> save 256-D cache

    Later runs:
      load .npz directly
    """

    pca_names = (
        pca_df["filename"]
        .astype(str)
        .tolist()
    )

    if os.path.exists(
        FEATURE_CACHE
    ):
        print(
            f"Loading cached 256-D features from {FEATURE_CACHE}"
        )

        cache = np.load(
            FEATURE_CACHE,
            allow_pickle=True,
        )

        names = cache[
            "filenames"
        ].astype(str)

        features = cache[
            "features"
        ].astype(
            np.float32
        )

        # Make sure the cache corresponds to current PCA file.
        if (
            len(names)
            == len(pca_names)
            and np.array_equal(
                names,
                np.asarray(
                    pca_names,
                    dtype=str,
                ),
            )
        ):
            print(
                f"Loaded cached features: {features.shape}"
            )

            return names, features

        print(
            "Feature cache does not match current PCA CSV; rebuilding."
        )

    extractor = (
        TrainedPenultimate256(
            trained_model
        )
        .to(DEVICE)
        .eval()
    )

    dataset = PCAImageDataset(
        pca_names
    )

    loader = DataLoader(
        dataset,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    features_list = []
    names_list = []

    print(
        f"Extracting trained 256-D features for "
        f"{len(dataset):,} PCA images..."
    )

    done = 0

    for images, names in loader:
        images = images.to(
            DEVICE
        )

        h = extractor(
            images
        )

        features_list.append(
            h.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        names_list.extend(
            list(names)
        )

        done += len(
            names
        )

        print(
            f"256-D cache: "
            f"{done}/{len(dataset)}"
        )

    features = np.concatenate(
        features_list,
        axis=0,
    )

    names_arr = np.asarray(
        names_list,
        dtype=str,
    )

    np.savez_compressed(
        FEATURE_CACHE,
        filenames=names_arr,
        features=features,
    )

    print(
        f"Saved 256-D feature cache: {FEATURE_CACHE}"
    )

    return names_arr, features


# ============================================================
# TRAINED FINAL LAYER AS ONE BINARY LOGIT
# ============================================================

class TrainedBinaryHead(nn.Module):
    """
    Softmax 2-class equivalence:

        z = logit_B - logit_A
          = (W_B-W_A) h + (b_B-b_A)
    """

    def __init__(
        self,
        final_linear,
    ):
        super().__init__()

        with torch.no_grad():
            w = (
                final_linear.weight[1]
                - final_linear.weight[0]
            ).detach().clone()

            b = (
                final_linear.bias[1]
                - final_linear.bias[0]
            ).detach().clone()

        self.weight = nn.Parameter(
            w
        )

        self.bias = nn.Parameter(
            b
        )

    def forward(
        self,
        h,
    ):
        return (
            h @ self.weight
            + self.bias
        )


def build_binary_head(
    model,
):
    head = TrainedBinaryHead(
        model.fc[4]
    ).to(DEVICE)

    if FREEZE_BIAS:
        head.bias.requires_grad_(
            False
        )

    return head


# ============================================================
# ONE JOINT UPDATE ON PCA-OPPOSITE A/B
# ============================================================

def one_pair_sgd_step(
    head,
    h_A,
    h_B,
    label_A,
    label_B,
):
    head.zero_grad(
        set_to_none=True
    )

    h_pair = torch.cat(
        [h_A, h_B],
        dim=0,
    )

    labels = torch.tensor(
        [
            float(label_A),
            float(label_B),
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    logits = head(
        h_pair
    )

    loss = (
        F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="mean",
        )
    )

    loss.backward()

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
            and head.bias.grad
            is not None
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

    return {
        "loss": float(
            loss.detach().cpu()
        ),

        "classification_grad_norm": (
            class_norm
        ),

        "l2_grad_norm": (
            l2_norm
        ),

        "class_to_l2_ratio": (
            class_norm
            / (l2_norm + 1e-12)
        ),

        "total_grad_norm": float(
            grad_total
            .norm()
            .cpu()
        ),
    }


# ============================================================
# GLOBAL PREDICTIONS ON PCA DATA
# ============================================================

@torch.no_grad()
def predict_all(
    head,
    all_features,
    batch_size=10000,
):
    preds_list = []
    probs_list = []

    for start in range(
        0,
        len(all_features),
        batch_size,
    ):
        h = torch.tensor(
            all_features[
                start:start + batch_size
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        probs_B = torch.sigmoid(
            head(h)
        )

        preds = (
            probs_B >= 0.5
        ).long()

        preds_list.append(
            preds.cpu().numpy()
        )

        probs_list.append(
            probs_B.cpu().numpy()
        )

    return (
        np.concatenate(
            preds_list
        ),
        np.concatenate(
            probs_list
        ),
    )


# ============================================================
# MODEL DIRECTION IN PCA
# ============================================================

def estimate_model_A_direction_pca(
    pca_df,
    preds,
):
    """
    Stable first-harmonic estimate of the classifier's orientation
    in the circular PCA plane.

    For each PCA point:
        predicted B -> +1
        predicted A -> -1

    The weighted circular vector estimates the direction of class B.
    The A-side direction is the opposite direction (+180 degrees).

    This is only a PCA visualization metric; the actual classifier
    lives in 256-D.
    """

    x = pca_df[
        "x"
    ].to_numpy(
        dtype=np.float64
    )

    y = pca_df[
        "y"
    ].to_numpy(
        dtype=np.float64
    )

    r = np.sqrt(
        x * x + y * y
    )

    valid = (
        r > 1e-12
    )

    ux = np.zeros_like(
        x
    )
    uy = np.zeros_like(
        y
    )

    ux[valid] = (
        x[valid]
        / r[valid]
    )

    uy[valid] = (
        y[valid]
        / r[valid]
    )

    sign = np.where(
        preds == 1,
        1.0,
        -1.0,
    )

    bx = np.sum(
        sign * ux
    )

    by = np.sum(
        sign * uy
    )

    if (
        abs(bx) < 1e-12
        and abs(by) < 1e-12
    ):
        return np.nan

    B_direction = (
        np.degrees(
            np.arctan2(
                by,
                bx,
            )
        )
        % 360
    )

    A_direction = (
        B_direction + 180.0
    ) % 360

    return float(
        A_direction
    )


# ============================================================
# PCA VISUALIZATION
# ============================================================

def save_pca_frame(
    pca_df,
    features,
    seq_df,
    iteration,
    head,
    rec,
):
    preds, _ = predict_all(
        head,
        features,
    )

    plot_df = pca_df.copy()
    plot_df["pred"] = preds

    df_A = plot_df[
        plot_df["pred"] == 0
    ]

    df_B = plot_df[
        plot_df["pred"] == 1
    ]

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.scatter(
        df_A["x"],
        df_A["y"],
        s=9,
        alpha=0.45,
        color="blue",
        label=(
            f"Predicted A "
            f"({len(df_A):,})"
        ),
    )

    ax.scatter(
        df_B["x"],
        df_B["y"],
        s=9,
        alpha=0.45,
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
        path["A_x"],
        path["A_y"],
        color="black",
        linewidth=1.5,
        label="A trajectory",
    )

    ax.scatter(
        [rec["A_x"]],
        [rec["A_y"]],
        marker="*",
        s=250,
        color="gold",
        edgecolors="black",
        zorder=10,
        label="Current A",
    )

    ax.scatter(
        [rec["B_x"]],
        [rec["B_y"]],
        marker="*",
        s=250,
        color="lime",
        edgecolors="black",
        zorder=10,
        label="Opposite B",
    )

    ax.plot(
        [
            rec["A_x"],
            rec["B_x"],
        ],
        [
            rec["A_y"],
            rec["B_y"],
        ],
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.7,
    )

    ax.scatter(
        [0],
        [0],
        marker="+",
        s=120,
        color="black",
        label="PCA origin",
    )

    lim = max(
        np.abs(
            pca_df[
                ["x", "y"]
            ].to_numpy()
        ).max(),
        TARGET_RADIUS,
    )

    lim *= 1.05

    ax.set_xlim(
        -lim,
        lim,
    )

    ax.set_ylim(
        -lim,
        lim,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(
        f"Iteration {iteration}\n"
        f"A P(B)={rec['prob_B_A_before']:.3f} | "
        f"B P(B)={rec['prob_B_B_before']:.3f} | "
        f"global A={rec['percent_A_all']:.1f}% "
        f"B={rec['percent_B_all']:.1f}%"
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"pca_boundary_{iteration:04d}.png",
        ),
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# SUMMARY PLOTS
# ============================================================

def save_summary_plots(
    df,
):
    # MATLAB-like angle tracking
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        df["iteration"],
        df["actual_angle_A"],
        label="A example",
    )

    ax.plot(
        df["iteration"],
        df["model_A_direction"],
        label="classifier A-side direction",
    )

    ax.set_xlabel("trial")
    ax.set_ylabel(
        "angle [degrees]"
    )

    ax.set_ylim(
        0,
        360,
    )

    ax.set_yticks(
        np.arange(
            0,
            361,
            45,
        )
    )

    ax.set_title(
        "PCA tracking: moving A example vs classifier"
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
            "angle_tracking_pca.png",
        ),
        dpi=200,
    )

    plt.close(fig)

    # Pair probabilities
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
        label="B image: P(B)",
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color="black",
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel("trial")
    ax.set_ylabel("P(B)")

    ax.set_title(
        "PCA pair predictions before update"
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

    # Global balance
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

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel("trial")
    ax.set_ylabel(
        "% of PCA images"
    )

    ax.set_title(
        "Global PCA classifier balance"
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
        "256-D final-layer weight norm"
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
        "Pair gradient vs L2"
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


# ============================================================
# MAIN
# ============================================================

def run():
    print(
        "Device:",
        DEVICE,
    )

    pca_df = load_pca()

    print(
        f"PCA images: {len(pca_df):,}"
    )

    seq_df = (
        generate_opposite_pca_sequence(
            pca_df
        )
    )

    model = (
        load_trained_model()
    )

    feature_names, features_256 = (
        extract_or_load_256_features(
            model,
            pca_df,
        )
    )

    feature_index = {
        filename: i
        for i, filename
        in enumerate(
            feature_names
        )
    }

    # Ensure every PCA row corresponds to the cached feature order.
    if not np.array_equal(
        feature_names.astype(str),
        pca_df[
            "filename"
        ].astype(str).to_numpy(),
    ):
        raise RuntimeError(
            "PCA rows and cached feature rows are not aligned."
        )

    head = build_binary_head(
        model
    )

    print(
        "Initial binary head "
        f"||w||={head.weight.detach().norm().item():.6f}"
    )

    initial_preds, _ = (
        predict_all(
            head,
            features_256,
        )
    )

    print(
        "Initial PCA classification: "
        f"A={(initial_preds == 0).mean()*100:.2f}% "
        f"B={(initial_preds == 1).mean()*100:.2f}%"
    )

    print()
    print(
        "Starting PCA opposite-pair experiment:"
    )

    if UNSUPERVISED:
        print(
            "A-side image + opposite PCA image "
            "-> own predictions -> one joint SGD+L2 step"
        )
    else:
        print(
            "A-side image=A + opposite PCA image=B "
            "-> one joint SGD+L2 step"
        )

    print()

    history = []

    for i, row in seq_df.iterrows():
        fn_A = row[
            "filename_A"
        ]

        fn_B = row[
            "filename_B"
        ]

        if (
            fn_A not in feature_index
            or fn_B not in feature_index
        ):
            print(
                f"Skipping iteration {i}: missing feature."
            )
            continue

        h_A = torch.tensor(
            features_256[
                feature_index[fn_A]
            ],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        h_B = torch.tensor(
            features_256[
                feature_index[fn_B]
            ],
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

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

        if UNSUPERVISED:
            label_A = int(
                prob_B_A_before
                >= 0.5
            )

            label_B = int(
                prob_B_B_before
                >= 0.5
            )
        else:
            label_A = 0
            label_B = 1

        step_info = (
            one_pair_sgd_step(
                head=head,
                h_A=h_A,
                h_B=h_B,
                label_A=label_A,
                label_B=label_B,
            )
        )

        all_preds, _ = (
            predict_all(
                head,
                features_256,
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

        model_A_direction = (
            estimate_model_A_direction_pca(
                pca_df,
                all_preds,
            )
        )

        rec = {
            **row.to_dict(),

            "prob_B_A_before": (
                prob_B_A_before
            ),

            "prob_B_B_before": (
                prob_B_B_before
            ),

            "label_A": label_A,
            "label_B": label_B,

            "model_A_direction": (
                model_A_direction
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

        history.append(
            rec
        )

        angle_text = (
            f"{model_A_direction:7.2f}°"
            if not np.isnan(
                model_A_direction
            )
            else "   NaN "
        )

        print(
            f"{i:4d} | "
            f"A={fn_A:12s} "
            f"P(B)={prob_B_A_before:.3f} | "
            f"B={fn_B:12s} "
            f"P(B)={prob_B_B_before:.3f} | "
            f"A-angle={row['actual_angle_A']:7.2f}° | "
            f"model-A={angle_text} | "
            f"global A={percent_A:6.2f}% "
            f"B={percent_B:6.2f}% | "
            f"ratio={rec['class_to_l2_ratio']:.3f}"
        )

        if (
            i == 0
            or i % SAVE_FRAME_EVERY == 0
            or i == len(seq_df) - 1
        ):
            save_pca_frame(
                pca_df=pca_df,
                features=features_256,
                seq_df=seq_df,
                iteration=i,
                head=head,
                rec=rec,
            )

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
    print("  pca_opposite_sequence.csv")
    print("  pca_boundary_XXXX.png")
    print("  angle_tracking_pca.png")
    print("  pair_probabilities.png")
    print("  global_class_balance.png")
    print("  weight_norm.png")
    print("  gradient_ratio.png")
    print("  online_learning_history.csv")


if __name__ == "__main__":
    run()
