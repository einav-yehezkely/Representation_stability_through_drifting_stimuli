import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


# ============================================================
# CONFIG
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
IMAGE_DIR = "female_faces"
MODEL_PATH = "model_ft_0_CE.pth"

OUTPUT_DIR = "output_pca_pair_fullmodel_adamw_fast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimizer: this is the setup that worked for supervised training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# PCA trajectory
NUM_STEPS = 360
ROTATION_RANGE_DEG = 360.0
START_ANGLE_DEG = 0.0
TARGET_RADIUS = 0.45

# Use each image only once across the generated sequence
UNIQUE_IMAGES = True

# Full expensive evaluation only every N online updates
EVAL_EVERY = 10

# Save a scatter frame whenever we perform a full evaluation
SAVE_FRAMES = True

# Evaluation batch size
EVAL_BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# PCA DATA
# ============================================================

def load_pca():
    df = pd.read_csv(
        PCA_CSV,
        header=None,
    )

    df.columns = [
        "filename",
        "x",
        "y",
    ]

    df["filename"] = (
        df["filename"]
        .astype(str)
    )

    df["angle_deg"] = (
        np.degrees(
            np.arctan2(
                df["y"],
                df["x"],
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
    return np.minimum(
        d,
        360.0 - d,
    )


def find_base_point(pca_df):
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

    idx = int(
        np.argmin(score)
    )

    return pca_df.loc[
        idx,
        ["x", "y"],
    ].to_numpy(
        dtype=np.float32
    )


def rotate_vector(v, angle_deg):
    a = np.deg2rad(
        angle_deg
    )

    R = np.array(
        [
            [
                np.cos(a),
                -np.sin(a),
            ],
            [
                np.sin(a),
                np.cos(a),
            ],
        ],
        dtype=np.float32,
    )

    return R @ v


# ============================================================
# BUILD OPPOSITE PCA PAIRS
# ============================================================

def generate_pair_sequence(pca_df):
    names = (
        pca_df["filename"]
        .to_numpy()
    )

    points = (
        pca_df[
            ["x", "y"]
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    base_point = find_base_point(
        pca_df
    )

    used_indices = set()
    rows = []

    for step in range(NUM_STEPS):
        rotation = (
            ROTATION_RANGE_DEG
            * step
            / NUM_STEPS
        )

        target_A = rotate_vector(
            base_point,
            rotation,
        )

        target_B = -target_A

        dists_A = np.linalg.norm(
            points - target_A,
            axis=1,
        )

        dists_B = np.linalg.norm(
            points - target_B,
            axis=1,
        )

        if UNIQUE_IMAGES and used_indices:
            used = np.fromiter(
                used_indices,
                dtype=np.int64,
            )

            dists_A[
                used
            ] = np.inf

            dists_B[
                used
            ] = np.inf

        idx_A = int(
            np.argmin(dists_A)
        )

        if UNIQUE_IMAGES:
            used_indices.add(
                idx_A
            )
            dists_B[
                idx_A
            ] = np.inf

        idx_B = int(
            np.argmin(dists_B)
        )

        if UNIQUE_IMAGES:
            used_indices.add(
                idx_B
            )

        point_A = points[
            idx_A
        ]

        point_B = points[
            idx_B
        ]

        actual_angle_A = (
            np.degrees(
                np.arctan2(
                    point_A[1],
                    point_A[0],
                )
            )
            % 360
        )

        actual_angle_B = (
            np.degrees(
                np.arctan2(
                    point_B[1],
                    point_B[0],
                )
            )
            % 360
        )

        rows.append(
            {
                "iteration": step,

                "filename_A": names[
                    idx_A
                ],

                "A_x": float(
                    point_A[0]
                ),

                "A_y": float(
                    point_A[1]
                ),

                "actual_angle_A": float(
                    actual_angle_A
                ),

                "filename_B": names[
                    idx_B
                ],

                "B_x": float(
                    point_B[0]
                ),

                "B_y": float(
                    point_B[1]
                ),

                "actual_angle_B": float(
                    actual_angle_B
                ),

                "distance_A_to_target": float(
                    dists_A[
                        idx_A
                    ]
                ),

                "distance_B_to_target": float(
                    dists_B[
                        idx_B
                    ]
                ),
            }
        )

    seq_df = pd.DataFrame(
        rows
    )

    seq_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pca_pair_sequence.csv",
        ),
        index=False,
    )

    print(
        f"Generated {len(seq_df)} opposite PCA pairs."
    )

    return seq_df


# ============================================================
# MODEL
# ============================================================

def load_model():
    model = (
        models.shufflenet_v2_x0_5(
            weights=None
        )
    )

    num_ftrs = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(
        nn.Dropout(
            p=0.5
        ),

        nn.Linear(
            num_ftrs,
            256,
        ),

        nn.ReLU(),

        nn.Dropout(
            p=0.3
        ),

        nn.Linear(
            256,
            2,
        ),
    )

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        state
    )

    model = model.to(
        DEVICE
    )

    return model


# ============================================================
# PRELOAD ALL PCA IMAGES ONCE
# ============================================================

def preload_pca_tensors(pca_df):
    """
    Opens every JPEG only once.

    Returns:
        all_tensors: [N, 3, 224, 224] CPU tensor
        valid_pca_df: PCA rows corresponding to all_tensors
        tensor_index: filename -> tensor row index
    """

    tensors = []
    valid_rows = []
    tensor_index = {}

    print()
    print(
        f"Preloading PCA images once: "
        f"{len(pca_df):,} requested..."
    )

    for _, row in pca_df.iterrows():
        filename = row[
            "filename"
        ]

        path = os.path.join(
            IMAGE_DIR,
            filename,
        )

        if not os.path.exists(
            path
        ):
            print(
                f"Warning: missing {path}"
            )
            continue

        img = Image.open(
            path
        ).convert("RGB")

        tensor = transform(
            img
        )

        tensor_index[
            filename
        ] = len(
            tensors
        )

        tensors.append(
            tensor
        )

        valid_rows.append(
            row
        )

        if (
            len(tensors) % 500
            == 0
        ):
            print(
                f"Preloaded "
                f"{len(tensors):,}/"
                f"{len(pca_df):,}"
            )

    if len(
        tensors
    ) == 0:
        raise RuntimeError(
            "No PCA images were loaded."
        )

    all_tensors = torch.stack(
        tensors,
        dim=0,
    )

    valid_pca_df = (
        pd.DataFrame(
            valid_rows
        )
        .reset_index(
            drop=True
        )
    )

    gb = (
        all_tensors.numel()
        * all_tensors.element_size()
        / 1024 ** 3
    )

    print(
        f"Preloaded {len(all_tensors):,} images "
        f"| RAM tensor size ≈ {gb:.2f} GB"
    )

    return (
        all_tensors,
        valid_pca_df,
        tensor_index,
    )


# ============================================================
# GET CURRENT PAIR FROM PRELOADED DATA
# ============================================================

def get_pair_batch(
    all_tensors,
    tensor_index,
    filename_A,
    filename_B,
):
    idx_A = tensor_index[
        filename_A
    ]

    idx_B = tensor_index[
        filename_B
    ]

    batch = torch.stack(
        [
            all_tensors[
                idx_A
            ],
            all_tensors[
                idx_B
            ],
        ],
        dim=0,
    )

    return batch.to(
        DEVICE
    )


# ============================================================
# ONE ADAMW STEP ON ALL LAYERS
# ============================================================

def one_full_model_step(
    model,
    optimizer,
    criterion,
    batch,
):
    model.train()

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    outputs = model(
        batch
    )

    probs_before = torch.softmax(
        outputs.detach(),
        dim=1,
    )

    loss = criterion(
        outputs,
        labels,
    )

    loss.backward()

    total_grad_sq = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total_grad_sq += float(
                torch.sum(
                    p.grad.detach()
                    ** 2
                ).cpu()
            )

    total_grad_norm = (
        total_grad_sq ** 0.5
    )

    # exactly ONE AdamW update
    optimizer.step()

    return {
        "loss": float(
            loss.detach().cpu()
        ),

        "prob_B_A_before": float(
            probs_before[
                0,
                1,
            ].cpu()
        ),

        "prob_B_B_before": float(
            probs_before[
                1,
                1,
            ].cpu()
        ),

        "total_grad_norm": (
            total_grad_norm
        ),
    }


# ============================================================
# FAST EVALUATION FROM PRELOADED TENSORS
# ============================================================

@torch.no_grad()
def classify_preloaded_images(
    model,
    all_tensors,
    batch_size=EVAL_BATCH_SIZE,
):
    model.eval()

    preds_all = []
    probs_B_all = []

    for start in range(
        0,
        len(all_tensors),
        batch_size,
    ):
        batch = all_tensors[
            start:start + batch_size
        ].to(
            DEVICE
        )

        outputs = model(
            batch
        )

        probs = torch.softmax(
            outputs,
            dim=1,
        )

        preds = torch.argmax(
            outputs,
            dim=1,
        )

        preds_all.append(
            preds.cpu().numpy()
        )

        probs_B_all.append(
            probs[:, 1]
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(
            preds_all
        ),
        np.concatenate(
            probs_B_all
        ),
    )


# ============================================================
# PCA CLASSIFIER DIRECTION
# ============================================================

def estimate_model_A_direction(
    pca_df,
    preds,
):
    x = (
        pca_df["x"]
        .to_numpy(
            dtype=np.float64
        )
    )

    y = (
        pca_df["y"]
        .to_numpy(
            dtype=np.float64
        )
    )

    r = np.sqrt(
        x * x
        + y * y
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

    ux[
        valid
    ] = (
        x[
            valid
        ]
        / r[
            valid
        ]
    )

    uy[
        valid
    ] = (
        y[
            valid
        ]
        / r[
            valid
        ]
    )

    # class B = +1, A = -1
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

    return float(
        (
            B_direction
            + 180.0
        )
        % 360
    )


# ============================================================
# FRAME
# ============================================================

def save_pca_frame(
    pca_df,
    preds,
    seq_df,
    iteration,
    rec,
):
    plot_df = (
        pca_df.copy()
    )

    plot_df[
        "pred"
    ] = preds

    df_A = plot_df[
        plot_df[
            "pred"
        ] == 0
    ]

    df_B = plot_df[
        plot_df[
            "pred"
        ] == 1
    ]

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    ax.scatter(
        df_A["x"],
        df_A["y"],
        s=8,
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
        s=8,
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
        path[
            "A_x"
        ],
        path[
            "A_y"
        ],
        linewidth=1.5,
        color="black",
        label="A trajectory",
    )

    ax.scatter(
        [
            rec[
                "A_x"
            ]
        ],
        [
            rec[
                "A_y"
            ]
        ],
        marker="*",
        s=250,
        color="gold",
        edgecolors="black",
        label="Current A",
        zorder=10,
    )

    ax.scatter(
        [
            rec[
                "B_x"
            ]
        ],
        [
            rec[
                "B_y"
            ]
        ],
        marker="*",
        s=250,
        color="lime",
        edgecolors="black",
        label="Current B",
        zorder=10,
    )

    ax.plot(
        [
            rec[
                "A_x"
            ],
            rec[
                "B_x"
            ],
        ],
        [
            rec[
                "A_y"
            ],
            rec[
                "B_y"
            ],
        ],
        linestyle="--",
        color="black",
        linewidth=1,
        alpha=0.6,
    )

    ax.scatter(
        [0],
        [0],
        marker="+",
        s=130,
        color="black",
        label="PCA origin",
    )

    lim = float(
        np.abs(
            pca_df[
                ["x", "y"]
            ].to_numpy()
        ).max()
        * 1.05
    )

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

    ax.set_xlabel(
        "PC1"
    )

    ax.set_ylabel(
        "PC2"
    )

    ax.set_title(
        f"Iteration {iteration}\n"
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
            f"pca_frame_{iteration:04d}.png",
        ),
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# SUMMARY PLOTS
# ============================================================

def save_summary_plots(
    history_df,
):
    # all iterations: pair probabilities
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        history_df[
            "iteration"
        ],
        history_df[
            "prob_B_A_before"
        ],
        label="A image: P(B)",
    )

    ax.plot(
        history_df[
            "iteration"
        ],
        history_df[
            "prob_B_B_before"
        ],
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

    ax.set_xlabel(
        "iteration"
    )

    ax.set_ylabel(
        "P(B)"
    )

    ax.set_title(
        "Pair predictions before each AdamW update"
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

    plt.close(
        fig
    )

    # loss every iteration
    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        history_df[
            "iteration"
        ],
        history_df[
            "loss"
        ],
    )

    ax.set_xlabel(
        "iteration"
    )

    ax.set_ylabel(
        "CrossEntropy loss"
    )

    ax.set_title(
        "Online pair loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            "loss.png",
        ),
        dpi=200,
    )

    plt.close(
        fig
    )

    # only full-evaluation rows
    eval_df = history_df.dropna(
        subset=[
            "model_A_direction"
        ]
    )

    if len(
        eval_df
    ) > 0:
        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            eval_df[
                "iteration"
            ],
            eval_df[
                "actual_angle_A"
            ],
            label="A example",
        )

        ax.plot(
            eval_df[
                "iteration"
            ],
            eval_df[
                "model_A_direction"
            ],
            label="classifier A direction",
        )

        ax.set_xlabel(
            "iteration"
        )

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
            "PCA tracking — full model AdamW"
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

        plt.close(
            fig
        )

        fig, ax = plt.subplots(
            figsize=(11, 5)
        )

        ax.plot(
            eval_df[
                "iteration"
            ],
            eval_df[
                "percent_A_all"
            ],
            label="% predicted A",
        )

        ax.plot(
            eval_df[
                "iteration"
            ],
            eval_df[
                "percent_B_all"
            ],
            label="% predicted B",
        )

        ax.set_ylim(
            0,
            100,
        )

        ax.set_xlabel(
            "iteration"
        )

        ax.set_ylabel(
            "% of PCA images"
        )

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

        plt.close(
            fig
        )


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
        f"PCA images: "
        f"{len(pca_df):,}"
    )

    seq_df = generate_pair_sequence(
        pca_df
    )

    # --------------------------------------------------------
    # Open / transform every PCA image once
    # --------------------------------------------------------
    (
        all_pca_tensors,
        eval_pca_df,
        tensor_index,
    ) = preload_pca_tensors(
        pca_df
    )

    # Sequence must use images that were actually loaded.
    missing = []

    for _, row in seq_df.iterrows():
        if (
            row["filename_A"]
            not in tensor_index
            or row["filename_B"]
            not in tensor_index
        ):
            missing.append(
                int(
                    row[
                        "iteration"
                    ]
                )
            )

    if missing:
        raise RuntimeError(
            f"Generated sequence contains missing images "
            f"at iterations: {missing[:20]}"
        )

    model = load_model()

    # All layers trainable
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    print()
    print(
        "Starting supervised online experiment:"
    )

    print(
        "PCA opposite pair -> "
        "one CE backward -> "
        "one AdamW step on ALL layers"
    )

    print(
        f"Full global evaluation every "
        f"{EVAL_EVERY} iterations."
    )

    print()

    history = []

    # keep last global metrics, useful for console only
    last_percent_A = np.nan
    last_percent_B = np.nan
    last_model_direction = np.nan

    for i, row in seq_df.iterrows():
        # ----------------------------------------------------
        # Fast pair construction: no JPEG open here
        # ----------------------------------------------------
        batch = get_pair_batch(
            all_tensors=all_pca_tensors,
            tensor_index=tensor_index,
            filename_A=row[
                "filename_A"
            ],
            filename_B=row[
                "filename_B"
            ],
        )

        # ----------------------------------------------------
        # Exactly one all-layer AdamW update
        # ----------------------------------------------------
        step_info = one_full_model_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            batch=batch,
        )

        # ----------------------------------------------------
        # Expensive global evaluation only occasionally
        # ----------------------------------------------------
        do_eval = (
            i == 0
            or i % EVAL_EVERY == 0
            or i == len(seq_df) - 1
        )

        preds_all = None

        if do_eval:
            preds_all, _ = classify_preloaded_images(
                model=model,
                all_tensors=all_pca_tensors,
                batch_size=EVAL_BATCH_SIZE,
            )

            last_percent_A = float(
                (
                    preds_all == 0
                ).mean()
                * 100
            )

            last_percent_B = float(
                (
                    preds_all == 1
                ).mean()
                * 100
            )

            last_model_direction = (
                estimate_model_A_direction(
                    eval_pca_df,
                    preds_all,
                )
            )

            percent_A_record = (
                last_percent_A
            )

            percent_B_record = (
                last_percent_B
            )

            direction_record = (
                last_model_direction
            )

        else:
            # NaN means "not evaluated at this iteration"
            percent_A_record = np.nan
            percent_B_record = np.nan
            direction_record = np.nan

        rec = {
            **row.to_dict(),

            **step_info,

            "percent_A_all": (
                percent_A_record
            ),

            "percent_B_all": (
                percent_B_record
            ),

            "model_A_direction": (
                direction_record
            ),

            "did_full_evaluation": (
                do_eval
            ),
        }

        history.append(
            rec
        )

        if do_eval:
            angle_text = (
                f"{last_model_direction:7.2f}°"
                if not np.isnan(
                    last_model_direction
                )
                else "NaN"
            )

            print(
                f"{i:4d} | "
                f"A={row['filename_A']:12s} "
                f"P(B)={step_info['prob_B_A_before']:.3f} | "
                f"B={row['filename_B']:12s} "
                f"P(B)={step_info['prob_B_B_before']:.3f} | "
                f"A-angle={row['actual_angle_A']:7.2f}° | "
                f"model-A={angle_text} | "
                f"global A={last_percent_A:6.2f}% "
                f"B={last_percent_B:6.2f}% | "
                f"loss={step_info['loss']:.5f}"
            )

            if SAVE_FRAMES:
                save_pca_frame(
                    pca_df=eval_pca_df,
                    preds=preds_all,
                    seq_df=seq_df,
                    iteration=i,
                    rec={
                        **rec,
                        "percent_A_all": (
                            last_percent_A
                        ),
                        "percent_B_all": (
                            last_percent_B
                        ),
                    },
                )

        else:
            print(
                f"{i:4d} | "
                f"A={row['filename_A']:12s} "
                f"P(B)={step_info['prob_B_A_before']:.3f} | "
                f"B={row['filename_B']:12s} "
                f"P(B)={step_info['prob_B_B_before']:.3f} | "
                f"A-angle={row['actual_angle_A']:7.2f}° | "
                f"loss={step_info['loss']:.5f}"
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
        model.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "model_after_online_pair_training.pth",
        ),
    )

    print()
    print(
        "Finished."
    )

    print(
        "Output directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    run()
