import os
import time
import shutil

from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from shufflenet_v2_x0_5_MSE_last_epoch import (
    get_dataloaders_from_lists,
    train_model_fast_for_self_training,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

BASE_DIR = "tmp_momentum_fast"
OUTPUT_DIR = "output_momentum_fast"

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_DIR = "female_faces"
PCA_CSV = "pca_top2_filtered_female.csv"
INITIAL_MODEL_PATH = "model_ft_0_MSE.pth"

ROTATION_DEGS = 0.5
NUM_ITERATIONS = 200
NUM_EPOCHS = 4
BATCH_SIZE_CLASSIFY = 50
BATCH_SIZE_TRAIN = 100
K_CLUSTER = 200
K_EVAL = 100

MOMENTUM_VALUES = [0.9]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PCA_DF = pd.read_csv(PCA_CSV, header=None)
PCA_DF.columns = ["filename", "x", "y"]
PCA_DF["angle_deg"] = np.degrees(np.arctan2(PCA_DF["y"], PCA_DF["x"])) % 360
ANGLE_MAP = dict(zip(PCA_DF["filename"], PCA_DF["angle_deg"]))


def inside_tmp(*paths):
    return os.path.join(BASE_DIR, *paths)


def inside_output(*paths):
    return os.path.join(OUTPUT_DIR, *paths)


# ─────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────

def load_top2_filtered(csv_path=PCA_CSV):
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    points = np.stack((x, y), axis=1)
    return names, points


def create_base_and_opposite_points(angle):
    names, points = load_top2_filtered(PCA_CSV)

    angles_deg = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
    radii = np.linalg.norm(points, axis=1)

    target_radius = 0.45
    angle_error = np.abs(angles_deg - angle)
    angle_error = np.minimum(angle_error, 360 - angle_error)
    radius_error = np.abs(radii - target_radius)

    combined_error = angle_error + radius_error * 100

    base_idx = np.argmin(combined_error)
    base_point = points[base_idx]
    opposite_point = -base_point

    return base_point, opposite_point


def rotate_vector(v, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)],
    ])
    return R @ v


def collect_nearest_images(center_point, all_points, all_names, output_name, k=200):
    """
    Fast version: does NOT copy images.
    Only finds nearest filenames and saves filenames_A.csv / filenames_B.csv.
    """
    dists = np.linalg.norm(all_points - center_point, axis=1)

    nearest_indices = np.argpartition(dists, k)[:k]
    nearest_indices = nearest_indices[np.argsort(dists[nearest_indices])]

    selected_names = [all_names[idx] for idx in nearest_indices]

    csv_path = inside_tmp(f"filenames_{output_name}.csv")
    pd.DataFrame(selected_names, columns=["filename"]).to_csv(csv_path, index=False)

    return nearest_indices


def merge_clusters():
    df_a = pd.read_csv(inside_tmp("filenames_A.csv"))
    df_b = pd.read_csv(inside_tmp("filenames_B.csv"))

    df_merged = pd.concat([df_a, df_b], ignore_index=True)
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    df_merged.to_csv(inside_tmp("filenames_merged.csv"), index=False)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def load_model(model_path=INITIAL_MODEL_PATH):
    model = models.shufflenet_v2_x0_5(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def update_teacher_ema(teacher, student, momentum):
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1 - momentum)


# ─────────────────────────────────────────────
# Batched classification
# ─────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def classify_images_batched(model, csv_path, clusters=False, batch_size=50):
    model.eval()
    df = pd.read_csv(csv_path)

    transform = get_transform()

    predicted_A = []
    predicted_B = []
    training_records = []

    batch_tensors = []
    batch_rows = []

    with torch.no_grad():
        for _, row in df.iterrows():
            image_path = os.path.join(IMAGE_DIR, row["filename"])

            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found. Skipping.")
                continue

            image = Image.open(image_path).convert("RGB")
            x = transform(image)

            batch_tensors.append(x)
            batch_rows.append(row)

            if len(batch_tensors) == batch_size:
                process_classification_batch(
                    model,
                    batch_tensors,
                    batch_rows,
                    predicted_A,
                    predicted_B,
                    training_records,
                    clusters,
                )
                batch_tensors = []
                batch_rows = []

        if batch_tensors:
            process_classification_batch(
                model,
                batch_tensors,
                batch_rows,
                predicted_A,
                predicted_B,
                training_records,
                clusters,
            )

    base_columns = df.columns.tolist()

    df_A = pd.DataFrame(predicted_A)
    df_B = pd.DataFrame(predicted_B)

    if df_A.empty:
        df_A = pd.DataFrame(columns=base_columns)
    else:
        df_A = df_A.reindex(columns=base_columns)

    if df_B.empty:
        df_B = pd.DataFrame(columns=base_columns)
    else:
        df_B = df_B.reindex(columns=base_columns)

    if clusters:
        df_A.to_csv(inside_tmp("cluster_predicted_as_A.csv"), index=False)
        df_B.to_csv(inside_tmp("cluster_predicted_as_B.csv"), index=False)
        return training_records

    df_A.to_csv(inside_tmp("predicted_as_A.csv"), index=False)
    df_B.to_csv(inside_tmp("predicted_as_B.csv"), index=False)
    return None


def process_classification_batch(
    model,
    batch_tensors,
    batch_rows,
    predicted_A,
    predicted_B,
    training_records,
    clusters,
):
    batch = torch.stack(batch_tensors).to(device)

    outputs = model(batch).squeeze(1)
    probs_b = torch.sigmoid(outputs)
    preds = (probs_b >= 0.5).long()

    for row, prob_b, pred in zip(batch_rows, probs_b, preds):
        prob_b = prob_b.item()
        prob_a = 1 - prob_b
        pred = pred.item()

        if pred == 0:
            predicted_A.append(row)
        else:
            predicted_B.append(row)

        if clusters:
            training_records.append({
                "filename": row["filename"],
                "prob_A": prob_a,
                "pred": "A" if pred == 0 else "B",
            })


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def take_k_closest_to_angle(filenames, center_angle_deg, k, pca_df=PCA_DF):
    df = pca_df[pca_df["filename"].isin(filenames)].copy()

    if df.empty:
        return []

    delta = (df["angle_deg"] - center_angle_deg).abs()
    df["ang_dist"] = np.minimum(delta, 360 - delta)

    return df.sort_values("ang_dist").head(k)["filename"].tolist()


def percent_predicted_as_filenames_batched(
    model,
    filenames,
    target_pred,
    batch_size=50,
):
    model.eval()
    transform = get_transform()

    count_target = 0
    total = 0
    batch_tensors = []

    with torch.no_grad():
        for fname in filenames:
            img_path = os.path.join(IMAGE_DIR, fname)

            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            x = transform(img)
            batch_tensors.append(x)

            if len(batch_tensors) == batch_size:
                count, n = predict_batch_count(model, batch_tensors, target_pred)
                count_target += count
                total += n
                batch_tensors = []

        if batch_tensors:
            count, n = predict_batch_count(model, batch_tensors, target_pred)
            count_target += count
            total += n

    percent = 100 * count_target / total if total > 0 else 0
    return percent, total


def predict_batch_count(model, batch_tensors, target_pred):
    batch = torch.stack(batch_tensors).to(device)

    outputs = model(batch).squeeze(1)
    probs = torch.sigmoid(outputs)
    preds = (probs >= 0.5).long()

    count = (preds == target_pred).sum().item()
    total = len(batch_tensors)

    return count, total


def compute_cluster_concentration(
    model,
    angle,
    iteration,
    cluster_concentration=None,
    k_eval=100,
):
    if cluster_concentration is None:
        cluster_concentration = []

    dfA = pd.read_csv(inside_tmp("filenames_A.csv"))
    dfB = pd.read_csv(inside_tmp("filenames_B.csv"))

    opposite_angle = (angle + 180) % 360

    eval_A_filenames = take_k_closest_to_angle(
        dfA["filename"].tolist(),
        angle,
        k_eval,
    )

    eval_B_filenames = take_k_closest_to_angle(
        dfB["filename"].tolist(),
        opposite_angle,
        k_eval,
    )

    pct_A_in_A, nA = percent_predicted_as_filenames_batched(
        model,
        eval_A_filenames,
        target_pred=0,
        batch_size=BATCH_SIZE_CLASSIFY,
    )

    pct_B_in_B, nB = percent_predicted_as_filenames_batched(
        model,
        eval_B_filenames,
        target_pred=1,
        batch_size=BATCH_SIZE_CLASSIFY,
    )

    cluster_concentration.append({
        "iteration": iteration,
        "angle": angle,
        "A_percent_in_A_cluster": pct_A_in_A,
        "B_percent_in_B_cluster": pct_B_in_B,
        "nA": nA,
        "nB": nB,
        "k_eval": k_eval,
        "opposite_angle": opposite_angle,
    })

    print(
        f"Iter {iteration}: A={pct_A_in_A:.1f}% near {angle:.1f}°, "
        f"B={pct_B_in_B:.1f}% near {opposite_angle:.1f}°"
    )

    return cluster_concentration


def plot_cluster_concentration(
    cluster_concentration,
    save_path,
    type_of_learning,
):
    df_seq = pd.DataFrame(cluster_concentration)

    if df_seq.empty:
        print("No cluster concentration data to plot.")
        return

    x = np.arange(len(df_seq))

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        df_seq["A_percent_in_A_cluster"],
        label="% predicted A in A cluster",
        linewidth=2,
    )

    plt.plot(
        x,
        df_seq["B_percent_in_B_cluster"],
        label="% predicted B in B cluster",
        linewidth=2,
    )

    step = max(1, len(x) // 12)
    angle_labels = [
        f"{row['angle']:.1f}°/{row['opposite_angle']:.1f}°"
        for _, row in df_seq.iterrows()
    ]

    plt.xticks(x[::step], angle_labels[::step], rotation=45)
    plt.xlabel("Base angle / opposite angle")
    plt.ylabel("% classified as intended label")
    plt.title(type_of_learning)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_momentum_comparison(results_per_momentum, save_path):
    plt.figure(figsize=(12, 6))

    for momentum, cluster_concentration in results_per_momentum.items():
        df = pd.DataFrame(cluster_concentration)

        if df.empty:
            continue

        x = np.arange(len(df))
        mean_tracking = (
            df["A_percent_in_A_cluster"] + df["B_percent_in_B_cluster"]
        ) / 2

        plt.plot(x, mean_tracking, label=f"m={momentum}", linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Mean tracking accuracy (%)")
    plt.title("Momentum comparison")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved comparison plot to {save_path}")


# ─────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────

def run_experiment(momentum_value, num_iterations=200):
    global BASE_DIR, OUTPUT_DIR

    m_str = str(momentum_value).replace(".", "_")
    BASE_DIR = f"tmp_momentum_fast_{m_str}"
    OUTPUT_DIR = f"output_momentum_fast_{m_str}"

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names, points = load_top2_filtered(PCA_CSV)
    base_point, opposite_point = create_base_and_opposite_points(0)

    student_model = load_model(INITIAL_MODEL_PATH)
    teacher_model = load_model(INITIAL_MODEL_PATH)
    teacher_model.eval()

    optimizer_ft = optim.AdamW(
        student_model.parameters(),
        lr=0.0001,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft,
        step_size=5,
        gamma=1,
    )

    cluster_concentration = []
    training_log = []

    start = time.time()

    for i in range(num_iterations):
        iter_start = time.time()
        angle_deg = (i * ROTATION_DEGS) % 360

        collect_nearest_images(
            base_point,
            points,
            names,
            output_name="A",
            k=K_CLUSTER,
        )

        collect_nearest_images(
            opposite_point,
            points,
            names,
            output_name="B",
            k=K_CLUSTER,
        )

        merge_clusters()

        # Teacher creates pseudo-labels
        t = time.time()
        training_records = classify_images_batched(
            teacher_model,
            inside_tmp("filenames_merged.csv"),
            clusters=True,
            batch_size=BATCH_SIZE_CLASSIFY,
        )
        print(f"[m={momentum_value}] classify time: {time.time() - t:.2f}s")

        for rec in training_records:
            training_log.append({
                "iteration": i,
                "cluster_angle": angle_deg,
                "filename": rec["filename"],
                "image_angle": ANGLE_MAP.get(rec["filename"], np.nan),
                "cluster_label": rec["pred"],
                "prob_A": rec["prob_A"],
            })

        df_A = pd.read_csv(inside_tmp("cluster_predicted_as_A.csv"))
        df_B = pd.read_csv(inside_tmp("cluster_predicted_as_B.csv"))

        print(f"[m={momentum_value}] Iter {i}: pseudo split A={len(df_A)}, B={len(df_B)}")

        filenames = df_A["filename"].tolist() + df_B["filename"].tolist()
        labels = [0] * len(df_A) + [1] * len(df_B)

        dataloaders, dataset_sizes, _ = get_dataloaders_from_lists(
            filenames,
            labels,
            image_dir=IMAGE_DIR,
            batch_size=BATCH_SIZE_TRAIN,
        )

        # Student trains fast
        student_model = train_model_fast_for_self_training(
            student_model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=NUM_EPOCHS,
        )

        # Teacher slowly follows student
        update_teacher_ema(teacher_model, student_model, momentum_value)

        # Evaluation
        t = time.time()
        cluster_concentration = compute_cluster_concentration(
            model=student_model,
            angle=angle_deg,
            iteration=i,
            cluster_concentration=cluster_concentration,
            k_eval=K_EVAL,
        )
        print(f"[m={momentum_value}] concentration time: {time.time() - t:.2f}s")

        # Save partial outputs every iteration
        pd.DataFrame(training_log).to_csv(
            inside_output("training_log_partial.csv"),
            index=False,
        )

        pd.DataFrame(cluster_concentration).to_csv(
            inside_output("cluster_concentration_partial.csv"),
            index=False,
        )

        base_point = rotate_vector(base_point, ROTATION_DEGS)
        opposite_point = rotate_vector(opposite_point, ROTATION_DEGS)

        cleanup_tmp_files()

        print(
            f"[m={momentum_value}] Iter {i} total time: "
            f"{time.time() - iter_start:.2f}s\n"
        )

    total_minutes = (time.time() - start) / 60
    print(f"[m={momentum_value}] Total time: {total_minutes:.1f} minutes")

    pd.DataFrame(training_log).to_csv(
        inside_output("training_log.csv"),
        index=False,
    )

    pd.DataFrame(cluster_concentration).to_csv(
        inside_output("cluster_concentration.csv"),
        index=False,
    )

    plot_cluster_concentration(
        cluster_concentration,
        save_path=inside_output("cluster_concentration_over_rotations.png"),
        type_of_learning=f"Momentum Encoder Fast (m={momentum_value})",
    )

    torch.save(student_model.state_dict(), inside_output("model_student_final.pth"))
    torch.save(teacher_model.state_dict(), inside_output("model_teacher_final.pth"))

    return cluster_concentration


def cleanup_tmp_files():
    files_to_delete = [
        inside_tmp("filenames_A.csv"),
        inside_tmp("filenames_B.csv"),
        inside_tmp("filenames_merged.csv"),
        inside_tmp("cluster_predicted_as_A.csv"),
        inside_tmp("cluster_predicted_as_B.csv"),
        inside_tmp("predicted_as_A.csv"),
        inside_tmp("predicted_as_B.csv"),
    ]

    for fname in files_to_delete:
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except Exception as e:
                print(f"Could not delete {fname}: {e}")


if __name__ == "__main__":
    print(f"Using device: {device}")

    all_results = {}

    for m in MOMENTUM_VALUES:
        print("\n" + "=" * 60)
        print(f"Running fast momentum experiment: m={m}")
        print("=" * 60 + "\n")

        result = run_experiment(
            momentum_value=m,
            num_iterations=NUM_ITERATIONS,
        )

        all_results[m] = result

    plot_momentum_comparison(
        all_results,
        save_path="momentum_fast_comparison.png",
    )

    print("\nAll fast momentum experiments done.")