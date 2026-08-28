import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# =========================
# Paths
# =========================

BASE_DIR = "tmp_perceptron_logic"
OUTPUT_DIR = "output_perceptron_logic"
IMAGE_DIR = "female_faces"
PCA_CSV = "pca_top2_filtered_female.csv"
MODEL_PATH = "model_ft_0_CE.pth"

SCATTER_DIR = os.path.join(OUTPUT_DIR, "scatter_frames")
LINEAR_DIR = os.path.join(OUTPUT_DIR, "linear_frames")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCATTER_DIR, exist_ok=True)
os.makedirs(LINEAR_DIR, exist_ok=True)


# =========================
# Save prints to output.txt
# =========================

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


log_file = open(os.path.join(OUTPUT_DIR, "output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


# =========================
# Parameters
# =========================

ROTATION_DEGS = 0.5
NUM_ITERATIONS = 720 * 3
LR = 1e-5
TARGET_RADIUS = 0.45
PLOT_EVERY = 10
WINDOW_SIZE = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =========================
# Data
# =========================

def load_pca(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["filename", "x", "y"]
    df["angle_deg"] = np.degrees(np.arctan2(df["y"], df["x"])) % 360
    points = df[["x", "y"]].values
    names = df["filename"].values
    return df, names, points


def create_start_point(angle_deg, df):
    points = df[["x", "y"]].values
    angles = df["angle_deg"].values
    radii = np.linalg.norm(points, axis=1)

    delta = np.abs(angles - angle_deg)
    angle_error = np.minimum(delta, 360 - delta)
    radius_error = np.abs(radii - TARGET_RADIUS)

    score = angle_error + 100 * radius_error
    idx = np.argmin(score)

    return points[idx]


def rotate_vector(v, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return R @ v


def get_nearest_image(center_point, all_points, all_names):
    dists = np.linalg.norm(all_points - center_point, axis=1)
    idx = np.argmin(dists)
    return all_names[idx], all_points[idx], dists[idx]


# =========================
# Model
# =========================

def load_model(model_path):
    model = models.shufflenet_v2_x0_5(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])


def load_image_tensor(filename):
    path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    return x


# =========================
# Perceptron-like self-training
# =========================

def one_self_training_step(model, optimizer, criterion, filename):
    model.eval()

    x = load_image_tensor(filename)

    with torch.no_grad():
        output_before = model(x)
        probs = torch.softmax(output_before, dim=1)
        pseudo_label = output_before.argmax(dim=1)
        prob_A = probs[0, 0].item()
        prob_B = probs[0, 1].item()
        confidence = probs.max(dim=1).values.item()

    model.train()
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, pseudo_label)

    loss.backward()
    optimizer.step()

    pred_str = "A" if pseudo_label.item() == 0 else "B"

    return {
        "pseudo_label": pseudo_label.item(),
        "pseudo_label_name": pred_str,
        "prob_A": prob_A,
        "prob_B": prob_B,
        "confidence": confidence,
        "loss": loss.item(),
    }


# =========================
# Evaluation
# =========================

def generate_rotation_sequence(base_point, all_points, all_names, num_steps=360):
    rows = []
    used = set()

    for i in range(num_steps):
        angle_deg = i
        center = rotate_vector(base_point, angle_deg)

        dists = np.linalg.norm(all_points - center, axis=1)

        for idx in used:
            dists[idx] = np.inf

        idx = np.argmin(dists)
        used.add(idx)

        true_angle = np.degrees(np.arctan2(center[1], center[0])) % 360

        rows.append({
            "step": i,
            "angle_deg": true_angle,
            "filename": all_names[idx],
        })

    return pd.DataFrame(rows)


def classify_sequence(model, seq_df):
    model.eval()
    rows = []

    with torch.no_grad():
        for _, row in seq_df.iterrows():
            filename = row["filename"]
            x = load_image_tensor(filename)

            output = model(x)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()

            rows.append({
                "step": row["step"],
                "angle_deg": row["angle_deg"],
                "filename": filename,
                "pred": "A" if pred == 0 else "B",
                "prob_A": probs[0, 0].item(),
                "prob_B": probs[0, 1].item(),
            })

    return pd.DataFrame(rows)


# =========================
# Graphs
# =========================
ANGLE_TRACKING_DIR = os.path.join(OUTPUT_DIR, "angle_tracking")
os.makedirs(ANGLE_TRACKING_DIR, exist_ok=True)


def create_angle_tracking_graph(training_log, rotation_degs, frame_id=None):
    """
    Graph like the MATLAB code:
    1. examples angle over trials
    2. weights/model angle approximation over trials
    3. norm/confidence-like radius graph
    """

    log_df = pd.DataFrame(training_log)

    trials = log_df["iteration"].values

    # examples angle: where the rotating stimulus should be
    example_angles = (trials * rotation_degs) % 360

    # weights angle approximation:
    # here we do not have real theta weights in PCA space,
    # so we use the model's current chosen/pseudo-labeled image angle as the tracked model angle
    weight_angles = log_df["image_angle"].values % 360

    plt.figure(figsize=(10, 5))
    plt.plot(trials, example_angles, label="examples")
    plt.plot(trials, weight_angles, "r", label="weights / model approximation")

    plt.xlabel("trial")
    plt.ylabel("angle")
    plt.legend()
    plt.title(f"change rate = {rotation_degs} degs/trial")
    plt.grid(True)
    plt.tight_layout()

    if frame_id is None:
        path = os.path.join(ANGLE_TRACKING_DIR, "angle_tracking.png")
    else:
        path = os.path.join(ANGLE_TRACKING_DIR, f"angle_tracking_{frame_id:03d}.png")

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved angle tracking graph: {path}")

    # Radius-like graph
    # In MATLAB this was norm(theta). Here the closest equivalent is confidence.
    plt.figure(figsize=(10, 4))
    plt.plot(trials, log_df["confidence"].values)

    plt.xlabel("trial")
    plt.ylabel("r / confidence")
    plt.title("Model confidence over trials")
    plt.grid(True)
    plt.tight_layout()

    if frame_id is None:
        path_r = os.path.join(ANGLE_TRACKING_DIR, "radius_tracking.png")
    else:
        path_r = os.path.join(ANGLE_TRACKING_DIR, f"radius_tracking_{frame_id:03d}.png")

    plt.savefig(path_r, dpi=300)
    plt.close()

    print(f"Saved radius tracking graph: {path_r}")
    
PERCEPTRON_GRAPH_DIR = os.path.join(OUTPUT_DIR, "perceptron_style_frames")
os.makedirs(PERCEPTRON_GRAPH_DIR, exist_ok=True)


def create_perceptron_style_graph(pca_df, pred_df, current_point, angle, frame_id):
    plot_df = pred_df.merge(pca_df, on="filename", how="left")

    df_A = plot_df[plot_df["pred"] == "A"]
    df_B = plot_df[plot_df["pred"] == "B"]

    # Estimate direction of A minus B in PCA space
    mean_A = df_A[["x", "y"]].mean().values if len(df_A) > 0 else np.array([0.0, 0.0])
    mean_B = df_B[["x", "y"]].mean().values if len(df_B) > 0 else np.array([0.0, 0.0])

    w = mean_A - mean_B

    if np.linalg.norm(w) < 1e-8:
        w = np.array([1.0, 0.0])

    w = w / np.linalg.norm(w)

    # Separator is perpendicular to w
    sep = np.array([-w[1], w[0]])

    radius = max(np.sqrt(pca_df["x"] ** 2 + pca_df["y"] ** 2)) * 1.1

    plt.figure(figsize=(8, 8))

    # Unit circle / PCA radius circle
    circle = plt.Circle((0, 0), radius, fill=False, color="black", linewidth=1)
    plt.gca().add_patch(circle)

    # Current rotating point
    plt.scatter(
        current_point[0],
        current_point[1],
        s=180,
        color="black",
        marker="o",
        label="Current point"
    )

    # Line from origin to current point
    plt.plot(
        [0, current_point[0]],
        [0, current_point[1]],
        color="black",
        linewidth=2,
        label="Rotating radius"
    )

    # Decision boundary
    plt.plot(
        [-radius * sep[0], radius * sep[0]],
        [-radius * sep[1], radius * sep[1]],
        color="red",
        linewidth=3,
        label="Estimated separating plane"
    )

    # Class direction vector
    plt.arrow(
        0, 0,
        radius * 0.7 * w[0],
        radius * 0.7 * w[1],
        head_width=0.03,
        head_length=0.04,
        length_includes_head=True,
        color="blue",
        linewidth=2
    )

    plt.text(
        radius * 0.75 * w[0],
        radius * 0.75 * w[1],
        "A direction",
        fontsize=12,
        color="blue"
    )

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)

    plt.title(f"Perceptron-style graph | iter {frame_id} | angle {angle:.1f}°")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        PERCEPTRON_GRAPH_DIR,
        f"perceptron_style_{frame_id:03d}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved perceptron-style graph: {save_path}")

def estimate_separator_angle(pred_df):
    df_sorted = pred_df.sort_values("angle_deg").reset_index(drop=True)

    transitions = []

    for i in range(len(df_sorted)):
        prev_i = (i - 1) % len(df_sorted)

        prev_pred = df_sorted.loc[prev_i, "pred"]
        curr_pred = df_sorted.loc[i, "pred"]

        if prev_pred != curr_pred:
            a1 = df_sorted.loc[prev_i, "angle_deg"]
            a2 = df_sorted.loc[i, "angle_deg"]

            # circular midpoint
            diff = ((a2 - a1 + 180) % 360) - 180
            mid = (a1 + diff / 2) % 360
            transitions.append(mid)

    if len(transitions) == 0:
        return None

    # line angle is modulo 180
    angles_rad = np.deg2rad(np.array(transitions) * 2)
    mean_angle = 0.5 * np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    return np.rad2deg(mean_angle) % 180

def create_perceptron_like_graph(pca_df, pred_df, current_filename, current_point, angle, frame_id):
    separator_angle = estimate_separator_angle(pred_df)

    plot_df = pred_df.merge(pca_df, on="filename", how="left")

    df_A = plot_df[plot_df["pred"] == "A"]
    df_B = plot_df[plot_df["pred"] == "B"]

    plt.figure(figsize=(9, 9))

    plt.scatter(pca_df["x"], pca_df["y"], s=5, alpha=0.2, color="gray", label="All images")
    plt.scatter(df_A["x"], df_A["y"], s=18, color="blue", label="Predicted A")
    plt.scatter(df_B["x"], df_B["y"], s=18, color="red", label="Predicted B")

    plt.scatter(
        current_point[0],
        current_point[1],
        s=180,
        color="black",
        marker="*",
        label="Current shaping point"
    )

    radius = max(np.sqrt(pca_df["x"] ** 2 + pca_df["y"] ** 2)) * 1.1

    if separator_angle is not None:
        rad = np.deg2rad(separator_angle)
        dx = radius * np.cos(rad)
        dy = radius * np.sin(rad)

        plt.plot(
            [-dx, dx],
            [-dy, dy],
            color="black",
            linewidth=3,
            linestyle="--",
            label=f"Empirical separator ~ {separator_angle:.1f}°"
        )

    # line from origin to current point
    plt.plot(
        [0, current_point[0]],
        [0, current_point[1]],
        color="black",
        linewidth=1,
        alpha=0.7
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    plt.title(f"Perceptron-like shaping | iter {frame_id} | point angle {angle:.1f}°")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "perceptron_like_frames")
    os.makedirs(path, exist_ok=True)

    save_path = os.path.join(path, f"perceptron_like_{frame_id:03d}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved perceptron-like graph: {save_path}")

def create_prediction_scatter(pca_df, pred_df, angle, frame_id):
    opposite_angle = (angle + 180) % 360

    df_A = pred_df[pred_df["pred"] == "A"]
    df_B = pred_df[pred_df["pred"] == "B"]

    df_A = df_A.merge(pca_df, on="filename", how="left")
    df_B = df_B.merge(pca_df, on="filename", how="left")

    plt.figure(figsize=(10, 10))

    plt.scatter(
        pca_df["x"],
        pca_df["y"],
        s=5,
        alpha=0.3,
        color="gray",
        label="All images"
    )

    plt.scatter(
        df_A["x"],
        df_A["y"],
        s=12,
        alpha=0.8,
        color="blue",
        label="Predicted A"
    )

    plt.scatter(
        df_B["x"],
        df_B["y"],
        s=12,
        alpha=0.8,
        color="red",
        label="Predicted B"
    )

    radius = max(np.sqrt(pca_df["x"] ** 2 + pca_df["y"] ** 2)) * 1.05
    circle = Circle((0, 0), radius, fill=False, color="black", linestyle="--", alpha=0.5)
    plt.gca().add_patch(circle)

    for angle_circ in range(0, 360, 20):
        rad = np.deg2rad(angle_circ)
        x = radius * np.cos(rad)
        y = radius * np.sin(rad)
        plt.plot([0, x], [0, y], color="gray", linewidth=0.5, alpha=0.5)
        plt.text(x * 1.05, y * 1.05, f"{angle_circ}°", ha="center", va="center")

    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)

    plt.title(
        f"Perceptron-like self-training\n"
        f"Iteration {frame_id}, current angle {angle:.1f}° / opposite {opposite_angle:.1f}°"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(SCATTER_DIR, f"scatter_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved scatter graph: {path}")


def create_linear_graph(pred_df, angle, frame_id, window_size=20):
    opposite_angle = (angle + 180) % 360

    results = []

    for step_angle in range(0, 360):
        end = (step_angle + window_size) % 360

        if step_angle < end:
            window_data = pred_df[
                (pred_df["angle_deg"] >= step_angle) &
                (pred_df["angle_deg"] < end)
            ]
        else:
            window_data = pred_df[
                (pred_df["angle_deg"] >= step_angle) |
                (pred_df["angle_deg"] < end)
            ]

        total = len(window_data)

        if total > 0:
            percent_A = (window_data["pred"] == "A").sum() * 100 / total
            percent_B = 100 - percent_A
        else:
            percent_A = 0
            percent_B = 0

        center_angle = (step_angle + window_size / 2) % 360

        results.append({
            "angle": center_angle,
            "percent_A": percent_A,
            "percent_B": percent_B,
        })

    df_results = pd.DataFrame(results)

    angle0 = df_results[df_results["angle"] == 0]
    angle360 = angle0.copy()
    angle360["angle"] = 360

    df_results = pd.concat([df_results, angle360], ignore_index=True)
    df_results = df_results.sort_values("angle")

    plt.figure(figsize=(12, 6))

    plt.plot(df_results["angle"], df_results["percent_A"], label="Predicted A", color="blue")
    plt.plot(df_results["angle"], df_results["percent_B"], label="Predicted B", color="red")

    plt.axvline(x=angle, color="blue", linewidth=1, linestyle="--", label="Current angle")
    plt.axvline(x=opposite_angle, color="red", linewidth=1, linestyle="--", label="Opposite angle")

    plt.xlabel("Angle")
    plt.ylabel("%")
    plt.title(
        f"Predictions along rotation sequence\n"
        f"Iteration {frame_id}, current angle {angle:.1f}°"
    )

    plt.ylim(0, 100)
    plt.xlim(0, 360)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(LINEAR_DIR, f"linear_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved linear graph: {path}")


# =========================
# Main
# =========================

if __name__ == "__main__":

    df, names, points = load_pca(PCA_CSV)

    base_point = create_start_point(angle_deg=0, df=df)

    model = load_model(MODEL_PATH)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    sequence_df = generate_rotation_sequence(
        base_point=base_point,
        all_points=points,
        all_names=names,
        num_steps=360,
    )

    sequence_df.to_csv(
        os.path.join(OUTPUT_DIR, "rotation_sequence_all.csv"),
        index=False
    )

    training_log = []
    start_time = time.time()
    current_point = base_point.copy()

    for i in range(NUM_ITERATIONS):
        angle_deg = (i * ROTATION_DEGS) % 360

        filename, chosen_point, dist = get_nearest_image(
            center_point=current_point,
            all_points=points,
            all_names=names,
        )

        result = one_self_training_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            filename=filename,
        )

        image_angle = df.loc[df["filename"] == filename, "angle_deg"].iloc[0]

        training_log.append({
            "iteration": i,
            "target_angle": angle_deg,
            "filename": filename,
            "image_angle": image_angle,
            "distance_from_center": dist,
            "pseudo_label": result["pseudo_label_name"],
            "prob_A": result["prob_A"],
            "prob_B": result["prob_B"],
            "confidence": result["confidence"],
            "loss": result["loss"],
        })

        print(
            f"Iter {i:03d} | "
            f"target angle={angle_deg:.2f}° | "
            f"image angle={image_angle:.2f}° | "
            f"pred={result['pseudo_label_name']} | "
            f"prob_A={result['prob_A']:.3f} | "
            f"prob_B={result['prob_B']:.3f} | "
            f"loss={result['loss']:.6f}"
        )

        if i % PLOT_EVERY == 0 and len(training_log) > 1:
            create_angle_tracking_graph(
                training_log=training_log,
                rotation_degs=ROTATION_DEGS,
                frame_id=i
            )

        current_point = rotate_vector(current_point, ROTATION_DEGS)

    end_time = time.time()

    pd.DataFrame(training_log).to_csv(
        os.path.join(OUTPUT_DIR, "training_log.csv"),
        index=False
    )

    create_angle_tracking_graph(
        training_log=training_log,
        rotation_degs=ROTATION_DEGS
    )

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "model_self_trained_perceptron_logic.pth")
    )

    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")

    log_file.close()