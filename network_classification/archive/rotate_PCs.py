import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Configuration
# ============================================================

PCA_CSV = "pca_top2_filtered_female.csv"
OUTPUT_DIR = "output_pca_linear_experiment"

NUM_ITERATIONS = 720
ROTATION_DEGS = 0.5

LEARNING_RATE = 0.1
WEIGHT_DECAY = 1.0

TARGET_RADIUS = 0.45

# True:
# Select the real image point nearest to the rotating target.
#
# False:
# Train directly on the ideal continuously rotating target point,
# exactly like the MATLAB experiment.
USE_REAL_PCA_POINTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
# Load PCA coordinates
# ============================================================

def load_pca_points(csv_path: str):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["filename", "PC1", "PC2"]

    points = df[["PC1", "PC2"]].to_numpy(dtype=np.float32)
    names = df["filename"].to_numpy()

    return df, names, points


# ============================================================
# Model
# ============================================================

class PCALinearClassifier(nn.Module):
    """
    Equivalent to the two-dimensional linear classifier in MATLAB:

        logit = theta_1 * PC1 + theta_2 * PC2

    There is no bias, so the separating boundary always passes
    through the origin.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ============================================================
# Geometry helpers
# ============================================================

def circular_difference_degrees(angle_a, angle_b):
    """
    Signed shortest angular difference angle_a - angle_b,
    in the range [-180, 180).
    """
    return (angle_a - angle_b + 180.0) % 360.0 - 180.0


def vector_angle_degrees(vector):
    return np.degrees(np.arctan2(vector[1], vector[0])) % 360.0


def select_nearest_real_point(
    target_point,
    all_points,
    all_names,
):
    distances = np.linalg.norm(all_points - target_point, axis=1)
    index = int(np.argmin(distances))

    return (
        all_points[index].copy(),
        all_names[index],
        float(distances[index]),
    )


# ============================================================
# Evaluation on a circle
# ============================================================

@torch.no_grad()
def evaluate_on_circle(model, radius=0.45, num_angles=720):
    angles_deg = np.linspace(
        0.0,
        360.0,
        num_angles,
        endpoint=False,
    )

    angles_rad = np.deg2rad(angles_deg)

    circle_points = np.column_stack(
        [
            radius * np.cos(angles_rad),
            radius * np.sin(angles_rad),
        ]
    ).astype(np.float32)

    inputs = torch.tensor(
        circle_points,
        dtype=torch.float32,
        device=DEVICE,
    )

    logits = model(inputs)
    probabilities = torch.sigmoid(logits)
    predictions = (logits >= 0).long()

    return (
        angles_deg,
        logits.cpu().numpy(),
        probabilities.cpu().numpy(),
        predictions.cpu().numpy(),
    )


# ============================================================
# Main experiment
# ============================================================

def run_experiment():
    _, names, points = load_pca_points(PCA_CSV)

    model = PCALinearClassifier().to(DEVICE)

    # Initial theta from the MATLAB code:
    #
    # rs = 0.114944784966984
    # psis = 1.569768424213780
    # theta = rs * [cos(psis), -sin(psis)]

    initial_radius = 0.114944784966984
    initial_angle = 1.569768424213780

    initial_theta = initial_radius * np.array(
        [
            np.cos(initial_angle),
            -np.sin(initial_angle),
        ],
        dtype=np.float32,
    )

    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(
                initial_theta.reshape(1, 2),
                dtype=torch.float32,
                device=DEVICE,
            )
        )

    # In PyTorch SGD, weight_decay adds:
    #
    #     WEIGHT_DECAY * theta
    #
    # to the gradient. Therefore with:
    #
    #     learning_rate = 0.1
    #     weight_decay = 1
    #
    # the old weights are multiplied approximately by 0.9,
    # matching the +theta term in the MATLAB gradient.

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        momentum=0.0,
    )

    criterion = nn.BCEWithLogitsLoss()

    history = []

    previous_filename = None

    for iteration in range(NUM_ITERATIONS):
        target_angle_deg = (
            iteration * ROTATION_DEGS
        ) % 360.0

        target_angle_rad = np.deg2rad(target_angle_deg)

        ideal_target_point = np.array(
            [
                TARGET_RADIUS * np.cos(target_angle_rad),
                TARGET_RADIUS * np.sin(target_angle_rad),
            ],
            dtype=np.float32,
        )

        if USE_REAL_PCA_POINTS:
            current_point, filename, selection_distance = (
                select_nearest_real_point(
                    target_point=ideal_target_point,
                    all_points=points,
                    all_names=names,
                )
            )
        else:
            current_point = ideal_target_point
            filename = "continuous_target"
            selection_distance = 0.0

        repeated_image = (
            filename == previous_filename
            if previous_filename is not None
            else False
        )
        previous_filename = filename

        x = torch.tensor(
            current_point,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        # ----------------------------------------------------
        # 1. Predict before training
        # ----------------------------------------------------

        with torch.no_grad():
            logit_before = model(x).item()
            probability_before = torch.sigmoid(
                torch.tensor(logit_before)
            ).item()

            pseudo_label = int(logit_before >= 0.0)

        y = torch.tensor(
            [float(pseudo_label)],
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # 2. One self-training update
        # ----------------------------------------------------

        optimizer.zero_grad()

        logit = model(x)
        loss = criterion(logit, y)

        loss.backward()
        optimizer.step()

        # ----------------------------------------------------
        # 3. Record updated model
        # ----------------------------------------------------

        with torch.no_grad():
            logit_after = model(x).item()

            theta = (
                model.linear.weight
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        theta_angle_deg = vector_angle_degrees(theta)

        # The line theta^T x = 0 is perpendicular to theta.
        boundary_angle_deg = (
            theta_angle_deg + 90.0
        ) % 360.0

        selected_point_angle_deg = vector_angle_degrees(
            current_point
        )

        weight_target_lag = circular_difference_degrees(
            theta_angle_deg,
            target_angle_deg,
        )

        history.append(
            {
                "iteration": iteration,
                "target_angle_deg": target_angle_deg,
                "selected_point_angle_deg": (
                    selected_point_angle_deg
                ),
                "filename": filename,
                "repeated_image": repeated_image,
                "selection_distance": selection_distance,
                "PC1": float(current_point[0]),
                "PC2": float(current_point[1]),
                "pseudo_label": pseudo_label,
                "logit_before": logit_before,
                "probability_class_1_before": (
                    probability_before
                ),
                "loss": float(loss.item()),
                "logit_after": logit_after,
                "theta_1": float(theta[0]),
                "theta_2": float(theta[1]),
                "theta_norm": float(np.linalg.norm(theta)),
                "theta_angle_deg": theta_angle_deg,
                "boundary_angle_deg": boundary_angle_deg,
                "weight_target_lag_deg": weight_target_lag,
            }
        )

        if iteration % 20 == 0:
            print(
                f"Iteration {iteration:4d} | "
                f"target={target_angle_deg:7.2f}° | "
                f"point={selected_point_angle_deg:7.2f}° | "
                f"theta={theta_angle_deg:7.2f}° | "
                f"label={pseudo_label} | "
                f"loss={loss.item():.5f} | "
                f"|theta|={np.linalg.norm(theta):.5f}"
            )

    history_df = pd.DataFrame(history)

    history_path = os.path.join(
        OUTPUT_DIR,
        "pca_tracking_history.csv",
    )
    history_df.to_csv(history_path, index=False)

    create_plots(model, history_df)

    torch.save(
        model.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "pca_linear_model.pth",
        ),
    )

    print(f"\nSaved results to: {OUTPUT_DIR}")


# ============================================================
# Plots
# ============================================================

def create_plots(model, history_df):
    iterations = history_df["iteration"]

    # --------------------------------------------------------
    # 1. Rotating point versus weight direction
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        iterations,
        history_df["target_angle_deg"],
        label="Rotating target angle",
    )

    plt.plot(
        iterations,
        history_df["selected_point_angle_deg"],
        label="Selected PCA point angle",
        alpha=0.7,
    )

    plt.plot(
        iterations,
        history_df["theta_angle_deg"],
        label="Weight-vector angle",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Angle in degrees")
    plt.title(
        "Tracking the Rotating PCA Point"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "angle_tracking.png",
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # 2. Angular lag
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        history_df["weight_target_lag_deg"],
    )

    plt.axhline(0.0, linewidth=1)

    plt.xlabel("Iteration")
    plt.ylabel("Weight angle − target angle")
    plt.title("Angular Tracking Lag")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "angular_lag.png",
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # 3. Weight norm
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        history_df["theta_norm"],
    )

    plt.xlabel("Iteration")
    plt.ylabel("Weight norm")
    plt.title("Norm of the Linear Classifier")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "weight_norm.png",
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # 4. Loss
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        iterations,
        history_df["loss"],
    )

    plt.xlabel("Iteration")
    plt.ylabel("BCE loss")
    plt.title("Self-Training Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "loss.png",
        ),
        dpi=300,
    )
    plt.close()

    # --------------------------------------------------------
    # 5. Final predictions around the full circle
    # --------------------------------------------------------

    (
        circle_angles,
        _,
        circle_probabilities,
        circle_predictions,
    ) = evaluate_on_circle(
        model,
        radius=TARGET_RADIUS,
        num_angles=720,
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        circle_angles,
        circle_probabilities,
        label="P(class 1)",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("PCA angle")
    plt.ylabel("Probability")
    plt.title(
        "Final Classifier Predictions Around the PCA Circle"
    )
    plt.xlim(0, 360)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "final_circle_probabilities.png",
        ),
        dpi=300,
    )
    plt.close()

    class_0_fraction = np.mean(
        circle_predictions == 0
    )
    class_1_fraction = np.mean(
        circle_predictions == 1
    )

    print(
        f"Final circle classification: "
        f"class 0 = {100 * class_0_fraction:.1f}%, "
        f"class 1 = {100 * class_1_fraction:.1f}%"
    )


if __name__ == "__main__":
    run_experiment()