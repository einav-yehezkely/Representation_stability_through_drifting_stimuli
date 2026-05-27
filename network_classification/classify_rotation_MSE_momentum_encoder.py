import os
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
from shufflenet_v2_x0_5_MSE import train_model, get_dataloaders, create_model_and_optim
import time
import torch.optim as optim
from torch.optim import lr_scheduler

BASE_DIR = "tmp_momentum_encoder"
os.makedirs(BASE_DIR, exist_ok=True)

OUTPUT_DIR = "output_momentum_encoder"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCA_DF = pd.read_csv("pca_top2_filtered_female.csv", header=None)
PCA_DF.columns = ["filename", "x", "y"]
PCA_DF["angle_deg"] = np.degrees(np.arctan2(PCA_DF["y"], PCA_DF["x"])) % 360
ANGLE_MAP = dict(zip(PCA_DF["filename"], PCA_DF["angle_deg"]))


def inside_tmp(*paths):
    """Return a path inside the BASE_DIR (tmp)."""
    return os.path.join(BASE_DIR, *paths)


def inside_output(*paths):
    """return a path inside the OUTPUT_DIR (output)."""
    return os.path.join(OUTPUT_DIR, *paths)


def load_top2_filtered(csv_path="pca_top2_filtered_female.csv"):
    """
    Load 2D PCA coordinates of pre-filtered images from a CSV file.

    Assumes the format: image_name, x, y

    The images in this file have been filtered such that they are not only close in the top 2 PCA components,
    but also have low variance in the remaining PCA dimensions (i.e., their radius in the residual components is small).
    This ensures that proximity in 2D reflects real similarity in the full FaceNet space.
    """
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    points = np.stack((x, y), axis=1)
    return names, points


def create_base_and_opposite_points(angle):
    # Load data
    names, points = load_top2_filtered("pca_top2_filtered_female.csv")

    # Compute angles (in radians) of each point from the origin
    angles = np.arctan2(points[:, 1], points[:, 0])

    # Convert angles from radians to degrees, now in range [-180, 180]
    angles_deg = np.degrees(angles)
    # Shift all angles to be in the range [0, 360)
    angles_deg = (angles_deg + 360) % 360

    radii = np.linalg.norm(points, axis=1)

    # Define the target angle in degrees
    target_angle = angle

    target_radius = 0.45

    angle_error = np.abs(angles_deg - target_angle)
    radius_error = np.abs(radii - target_radius)
    combined_error = angle_error + radius_error * 100

    # Find the index of the point whose angle is closest to the target angle
    base_idx = np.argmin(combined_error)
    # Retrieve the actual 2D PCA coordinates of the selected base point
    base_point = points[base_idx]

    opposite_point = -base_point

    return base_point, opposite_point


def rotate_vector(v, angle_deg):
    """
    Rotate a 2D vector counter clockwise by angle_deg (in degrees) around the origin.

    Parameters:
    - v: numpy array of shape (2,), the vector to rotate
    - angle_deg: float, the rotation angle in degrees

    Returns:
    - rotated vector (2D numpy array)
    """
    angle_rad = np.deg2rad(angle_deg)
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return R @ v


def collect_nearest_images(
    center_point,
    all_points,
    all_names,
    output_dir,
    k=500,
    image_source_dir="female_faces",
):
    """
    Find the k nearest images to center_point and copy them to output_dir.
    Also saves a CSV file with the selected filenames.

    Parameters:
        center_point: np.array of shape (2,)
        all_points: np.array of shape (N, 2)
        all_names: list or array of N image filenames
        output_dir: path to the folder where results will be saved
        k: number of images to select
        image_source_dir: directory where source images are located
    """
    # If the output directory already exists, delete all its contents
    if os.path.exists(output_dir):
        # Iterate through all files and folders in the output directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                # If it's a file or symbolic link, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # If it's a directory, delete it and all its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                # If something goes wrong, print a warning message
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # If the directory does not exist, create it
        os.makedirs(output_dir)

    # Compute distances
    dists = np.linalg.norm(all_points - center_point, axis=1)
    nearest_indices = np.argsort(dists)[:k]

    selected_names = []

    for idx in nearest_indices:
        name = all_names[idx]
        src_path = os.path.join(image_source_dir, name)
        dst_path = os.path.join(output_dir, name)
        selected_names.append(name)
        try:
            shutil.copy2(src_path, dst_path)
        except FileNotFoundError:
            print(f"Warning: {src_path} not found.")

    # Save filenames to CSV
    csv_name = f"filenames_{os.path.basename(output_dir)}.csv"
    csv_path = inside_tmp(csv_name)
    pd.DataFrame(selected_names, columns=["filename"]).to_csv(csv_path, index=False)
    print(f"Saved {len(selected_names)} image names to {csv_path}")

    return nearest_indices


def merge_clusters():
    """
    Merge the two CSV files filenames_A.csv and filenames_B.csv into filenames_merged.csv in order to retrain the model
    on both clusters.
    """
    # Load both CSVs
    df_a = pd.read_csv(inside_tmp("filenames_A.csv"))
    df_b = pd.read_csv(inside_tmp("filenames_B.csv"))

    # Concatenate the DataFrames
    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    # Shuffle rows
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to new CSV
    df_merged.to_csv(inside_tmp("filenames_merged.csv"), index=False)


def load_model(model_path="model_ft_0_MSE.pth"):
    """
    Load the pre-trained model for classification.
    The model is trained on images from 135 and 315 degrees.
    """
    model = models.shufflenet_v2_x0_5(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        # nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        # nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def classify_images(model, csv_path, clusters=False):
    model.eval()
    # Load CSV
    df = pd.read_csv(csv_path)

    # Image transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    predicted_A = []
    predicted_B = []
    training_records = []

    for i, row in df.iterrows():
        image_path = os.path.join("female_faces", row["filename"])

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob_b = torch.sigmoid(output).item()
            prob_a = 1 - prob_b
            pred = 1 if prob_b >= 0.5 else 0

        # Track predictions
        if pred == 0:
            predicted_A.append(row)
        else:
            predicted_B.append(row)

        if clusters:
            training_records.append(
                {
                    "filename": row["filename"],
                    "prob_A": prob_a,
                    "pred": "A" if pred == 0 else "B",
                }
            )

    # Save predicted CSVs safely, even if one list is empty
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
    else:
        df_A.to_csv(inside_tmp("predicted_as_A.csv"), index=False)
        df_B.to_csv(inside_tmp("predicted_as_B.csv"), index=False)


def split_and_copy_images(
    csv_path,
    label,
    image_source_dir="female_faces",
    train_ratio=0.8,
    root_dir=None,  # None → computed at call time from current BASE_DIR
):
    if root_dir is None:
        root_dir = inside_tmp("split_data")

    # Load the CSV with predicted filenames
    df = pd.read_csv(csv_path)
    filenames = df["filename"].tolist()

    # Split into train and val sets
    train_files, val_files = train_test_split(
        filenames, train_size=train_ratio, random_state=42
    )

    # Define output directories
    for subset, files in [("train", train_files), ("val", val_files)]:
        target_dir = os.path.join(root_dir, subset, label)
        os.makedirs(target_dir, exist_ok=True)

        for name in tqdm(files, desc=f"Copying {subset}/{label}"):
            src = os.path.join(image_source_dir, name)
            dst = os.path.join(target_dir, name)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Warning: {src} not found.")


def generate_rotation_sequence(
    base_point,
    all_points,
    all_names,
    num_steps=180,
    start_angle=0,
    rotation_range=180,
    used_indices=None,
):
    """
    Rotate base_point around the origin in num_steps steps (in degrees)
    and find the closest point from all_points at each step.

    Parameters:
    - base_point: 2D numpy array representing the starting point
    - all_points: numpy array of shape (N, 2), 2D positions of all images
    - all_names: list or array of N image names corresponding to all_points
    - num_steps: number of rotation steps (default 1000 = every 0.36 degrees)
    - start_angle: starting angle in degrees (default 0)
    - rotation_range: range of rotation in degrees (default 180)
    - used_indices: set of indices to avoid reusing across runs

    Returns:
    - List of tuples: (step_index, angle_in_degrees, closest_image_name)
    """
    results = []
    if used_indices is None:
        used_indices = set()

    for i in range(num_steps):
        angle_deg = (start_angle + (rotation_range * i / num_steps)) % 360
        rotated = rotate_vector(base_point, angle_deg)
        true_angle = np.degrees(np.arctan2(rotated[1], rotated[0])) % 360
        dists = np.linalg.norm(all_points - rotated, axis=1)

        for idx in used_indices:
            dists[idx] = np.inf  # Ignore already used indices

        idx_closest = np.argmin(dists)
        used_indices.add(idx_closest)
        results.append((i, true_angle, all_names[idx_closest]))
    return results, used_indices


def take_k_closest_to_angle(filenames, center_angle_deg, k, pca_df=PCA_DF):
    """
    filenames: iterable of image filenames
    center_angle_deg: angle in [0,360)
    returns: list of k filenames closest in ANGLE to center_angle_deg (circular distance)
    """
    df = pca_df[pca_df["filename"].isin(filenames)].copy()
    if df.empty:
        return []

    delta = (df["angle_deg"] - center_angle_deg).abs()
    df["ang_dist"] = np.minimum(delta, 360 - delta)

    return df.sort_values("ang_dist").head(k)["filename"].tolist()


def percent_predicted_as_filenames(
    model, filenames, target_pred, image_source_dir="female_faces"
):
    """
    target_pred: 0 for A, 1 for B
    Returns: (percent, n_used)
    """
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    count_target = 0
    total = 0

    for fname in filenames:
        img_path = os.path.join(image_source_dir, fname)
        if not os.path.exists(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x)
            prob_b = torch.sigmoid(output).item()
            pred = 1 if prob_b >= 0.5 else 0

        count_target += int(pred == target_pred)
        total += 1

    percent = (100 * count_target / total) if total > 0 else 0
    return percent, total


def compute_cluster_concentration(
    angle, iteration, cluster_concentration=None, k_eval=100, model=None
):
    """
    Evaluate the current trained classifier after clustering-based training.

    Measures:
      - Among the k_eval images closest to the current A angle,
        what percentage is classified as A by the final sigmoid classifier.
      - Among the k_eval images closest to the opposite B angle,
        what percentage is classified as B by the final sigmoid classifier.

    model: the model to evaluate. Falls back to the global self_training_model if None.
    """
    if cluster_concentration is None:
        cluster_concentration = []

    if model is None:
        raise ValueError("model parameter is required in this file.")
    eval_model = model

    a_csv = inside_tmp("filenames_A.csv")  # 1000
    b_csv = inside_tmp("filenames_B.csv")  # 1000

    if not (os.path.exists(a_csv) and os.path.exists(b_csv)):
        print("Warning: filenames_A/B.csv not found. Skipping.")
        return cluster_concentration

    dfA = pd.read_csv(a_csv)
    dfB = pd.read_csv(b_csv)

    opposite_angle = (angle + 180) % 360

    # pick only k_eval closest-by-angle within each 1000 cluster
    eval_A_filenames = take_k_closest_to_angle(dfA["filename"].tolist(), angle, k_eval)
    eval_B_filenames = take_k_closest_to_angle(
        dfB["filename"].tolist(), opposite_angle, k_eval
    )

    pct_A_in_A, nA = percent_predicted_as_filenames(
        eval_model, eval_A_filenames, target_pred=0
    )
    pct_B_in_B, nB = percent_predicted_as_filenames(
        eval_model, eval_B_filenames, target_pred=1
    )

    cluster_concentration.append(
        {
            "iteration": iteration,
            "angle": angle,
            "A_percent_in_A_cluster": pct_A_in_A,
            "B_percent_in_B_cluster": pct_B_in_B,
            "nA": nA,
            "nB": nB,
            "k_eval": k_eval,
            "opposite_angle": opposite_angle,
        }
    )

    print(
        f"Iter {iteration}: classifier predicts A for {pct_A_in_A:.1f}% "
        f"of {k_eval} images near {angle:.1f}° (n={nA}), "
        f"and predicts B for {pct_B_in_B:.1f}% "
        f"of {k_eval} images near {opposite_angle:.1f}° (n={nB})"
    )

    return cluster_concentration


def plot_cluster_concentration(
    cluster_concentration,
    save_path="cluster_concentration_over_rotations.png",
    type_of_learning="Unsupervised Learning",
):
    """
    Plot:
      - % predicted as A in cluster A (1000 images around base angle)
      - % predicted as B in cluster B (1000 images around opposite angle)

    X-axis stays in rotation order, but tick labels show: base_angle/opposite_angle
    """
    df_seq = pd.DataFrame(cluster_concentration)

    x = np.arange(len(df_seq))

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        df_seq["A_percent_in_A_cluster"],
        label="% predicted A in cluster centered at base angle",
        color="blue",
        linewidth=2,
    )

    plt.plot(
        x,
        df_seq["B_percent_in_B_cluster"],
        label="% predicted B in cluster centered at opposite angle",
        color="red",
        linewidth=2,
    )

    angle_labels = [
        f"{row['angle']:.2f}°/{((row['angle'] + 180) % 360):.2f}°"
        for _, row in df_seq.iterrows()
    ]

    step = max(1, len(x) // 12)
    plt.xticks(x[::step], angle_labels[::step], rotation=45)

    plt.xlabel("Base angle / Opposite angle")
    plt.ylabel("% of images classified as cluster label")

    plt.title(
        f"At Each Rotation Step: How Many of the 100 Closest Images\n"
        f"Are Classified as the Cluster's Intended Label\n"
        f"-- {type_of_learning} --"
    )

    plt.ylim(0, 100)
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_label_change_csvs(training_log_path="training_log.csv"):
    df = pd.read_csv(training_log_path)

    df = df.sort_values(["filename", "iteration"]).reset_index(drop=True)

    # Convert cluster labels to binary values:
    # A -> 0, B -> 1
    df["pred_binary"] = (df["cluster_label"] == "B").astype(int)

    df["prev_iteration"] = df.groupby("filename")["iteration"].shift(1)
    df["prev_cluster_label"] = df.groupby("filename")["cluster_label"].shift(1)
    df["prev_pred_binary"] = df.groupby("filename")["pred_binary"].shift(1)
    df["prev_cluster_angle"] = df.groupby("filename")["cluster_angle"].shift(1)

    df["is_consecutive"] = df["iteration"] == df["prev_iteration"] + 1

    changed = df[
        df["prev_iteration"].notna()
        & df["is_consecutive"]
        & (df["pred_binary"] != df["prev_pred_binary"])
    ].copy()

    changed["from_label"] = changed["prev_cluster_label"]
    changed["to_label"] = changed["cluster_label"]
    changed["from_iteration"] = changed["prev_iteration"].astype(int)
    changed["to_iteration"] = changed["iteration"].astype(int)

    changed_csv = changed[
        [
            "filename",
            "image_angle",
            "from_iteration",
            "to_iteration",
            "from_label",
            "to_label",
            "prev_cluster_angle",
            "cluster_angle",
        ]
    ].rename(
        columns={
            "prev_cluster_angle": "from_cluster_angle",
            "cluster_angle": "to_cluster_angle",
        }
    )

    changed_csv.to_csv(
        inside_output("changed_images_between_consecutive_iterations.csv"),
        index=False,
    )

    summary = (
        changed_csv.groupby("to_iteration")["filename"]
        .count()
        .reset_index(name="num_changed_images")
    )

    summary.to_csv(
        inside_output("changed_images_summary_per_iteration.csv"),
        index=False,
    )

    return changed_csv, summary


def compute_cluster_classification_errors(model, iteration, angle):
    """
    Counts how many images in each 200-image cluster
    are classified incorrectly BEFORE retraining.
    """

    model.eval()

    dfA = pd.read_csv(inside_tmp("filenames_A.csv"))
    dfB = pd.read_csv(inside_tmp("filenames_B.csv"))

    filenames_A = dfA["filename"].tolist()
    filenames_B = dfB["filename"].tolist()

    correct_A, nA = percent_predicted_as_filenames(
        model,
        filenames_A,
        target_pred=0,
    )

    correct_B, nB = percent_predicted_as_filenames(
        model,
        filenames_B,
        target_pred=1,
    )

    correct_A_count = round(correct_A * nA / 100)
    correct_B_count = round(correct_B * nB / 100)

    wrong_A_count = nA - correct_A_count
    wrong_B_count = nB - correct_B_count

    return {
        "iteration": iteration,
        "angle": angle,
        "wrong_A_count": wrong_A_count,
        "wrong_B_count": wrong_B_count,
        "correct_A_count": correct_A_count,
        "correct_B_count": correct_B_count,
    }


# ─────────────────────────────────────────────
# Momentum encoder
# ─────────────────────────────────────────────

def update_teacher_ema(teacher, student, momentum):
    """
    Update teacher weights as exponential moving average of student weights.

    momentum=0   → teacher is replaced by student each step (no memory, pure self-training)
    momentum→1   → teacher changes very slowly (strong temporal memory)
    """
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data


def run_experiment(momentum_value, num_iterations=720):
    """
    Run one full rotation experiment with a given EMA momentum value.

    The teacher generates pseudo-labels; the student trains on them.
    After each training step the teacher is updated via EMA of the student.
    """
    global BASE_DIR, OUTPUT_DIR

    m_str = str(momentum_value).replace(".", "_")
    BASE_DIR = f"tmp_momentum_{m_str}"
    OUTPUT_DIR = f"output_momentum_{m_str}"

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    names, points = load_top2_filtered("pca_top2_filtered_female.csv")
    base_point, opposite_point = create_base_and_opposite_points(0)

    student_model = load_model(model_path="model_ft_0_MSE.pth").to(device)
    teacher_model = load_model(model_path="model_ft_0_MSE.pth").to(device)
    teacher_model.eval()

    optimizer_ft = optim.AdamW(student_model.parameters(), lr=0.0001, weight_decay=1e-4)

    rotation_seq, _ = generate_rotation_sequence(
        base_point=base_point,
        all_points=points,
        all_names=names,
        num_steps=360,
        start_angle=0,
        rotation_range=360,
        used_indices=set(),
    )

    df_seq = pd.DataFrame(rotation_seq, columns=["step", "angle_deg", "filename"])
    df_seq.to_csv(inside_tmp("rotation_sequence_all.csv"), index=False)
    print(f"[m={momentum_value}] Generated rotation sequence.\n")

    cluster_concentration = []
    training_log = []
    classification_error_log = []

    start = time.time()
    for i in range(num_iterations):
        collect_nearest_images(base_point, points, names, output_dir=inside_tmp("A"), k=200)
        collect_nearest_images(opposite_point, points, names, output_dir=inside_tmp("B"), k=200)

        merge_clusters()

        # TEACHER generates pseudo-labels
        training_records = classify_images(
            teacher_model, inside_tmp("filenames_merged.csv"), clusters=True
        )

        angle_deg = (i * 0.5) % 360

        for rec in training_records:
            training_log.append(
                {
                    "iteration": i,
                    "cluster_angle": angle_deg,
                    "filename": rec["filename"],
                    "image_angle": ANGLE_MAP.get(rec["filename"], np.nan),
                    "cluster_label": rec["pred"],
                    "prob_A": rec["prob_A"],
                }
            )

        error_info = compute_cluster_classification_errors(
            teacher_model, iteration=i, angle=angle_deg
        )
        classification_error_log.append(error_info)
        print(
            f"[m={momentum_value}] Iter {i}: "
            f"A wrong = {error_info['wrong_A_count']}/200 | "
            f"B wrong = {error_info['wrong_B_count']}/200"
        )

        split_and_copy_images(inside_tmp("cluster_predicted_as_A.csv"), label="A")
        split_and_copy_images(inside_tmp("cluster_predicted_as_B.csv"), label="B")

        dataloaders, dataset_sizes, _ = get_dataloaders(data_dir=inside_tmp("split_data"))

        criterion = nn.MSELoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=1)

        # STUDENT trains on teacher's labels
        student_model = train_model(
            student_model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=4,
            plots=False,
        )

        # Update TEACHER with EMA
        update_teacher_ema(teacher_model, student_model, momentum_value)

        # Measure how well the student tracks the current cluster positions
        cluster_concentration = compute_cluster_concentration(
            angle=angle_deg,
            iteration=i,
            cluster_concentration=cluster_concentration,
            model=student_model,
        )

        base_point = rotate_vector(base_point, angle_deg=0.5)
        opposite_point = rotate_vector(opposite_point, angle_deg=0.5)

        shutil.rmtree(inside_tmp("A"), ignore_errors=True)
        shutil.rmtree(inside_tmp("B"), ignore_errors=True)
        shutil.rmtree(inside_tmp("split_data"), ignore_errors=True)

        for fname in [
            inside_tmp("filenames_A.csv"),
            inside_tmp("filenames_B.csv"),
            inside_tmp("predicted_as_A.csv"),
            inside_tmp("predicted_as_B.csv"),
            inside_tmp("filenames_merged.csv"),
            inside_tmp("cluster_predicted_as_A.csv"),
            inside_tmp("cluster_predicted_as_B.csv"),
        ]:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except Exception as e:
                    print(f"Could not delete {fname}: {e}")

    end = time.time()
    print(f"[m={momentum_value}] Total time: {(end - start) / 60:.1f} minutes")

    plot_cluster_concentration(
        cluster_concentration,
        save_path=inside_output("cluster_concentration_over_rotations.png"),
        type_of_learning=f"Momentum Encoder (m={momentum_value})",
    )

    pd.DataFrame(training_log).to_csv(inside_output("training_log.csv"), index=False)

    df_errors = pd.DataFrame(classification_error_log)
    df_errors.to_csv(inside_output("cluster_classification_errors.csv"), index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_errors["iteration"], df_errors["wrong_A_count"],
        marker="o", label="Wrong in cluster A",
    )
    plt.plot(
        df_errors["iteration"], df_errors["wrong_B_count"],
        marker="o", label="Wrong in cluster B",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Number of incorrectly classified images")
    plt.title(
        f"Wrongly Classified Images in the 200 Closest Images\n"
        f"Before Self-Training — Momentum Encoder (m={momentum_value})"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(inside_output("cluster_classification_errors.png"), dpi=300)
    plt.close()

    torch.save(student_model.state_dict(), inside_output("model_student_final.pth"))
    torch.save(teacher_model.state_dict(), inside_output("model_teacher_final.pth"))

    return cluster_concentration


def plot_momentum_comparison(results_per_momentum, save_path="momentum_comparison.png"):
    """
    Compare rotation-tracking accuracy across momentum values.
    Two subplots: A-cluster accuracy and B-cluster accuracy over iterations.
    """
    colors = {0: "black", 0.9: "green", 0.99: "orange", 0.999: "purple"}

    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    n_iters = len(next(iter(results_per_momentum.values())))
    tick_positions = np.linspace(0, n_iters - 1, 13, dtype=int)
    tick_labels = [f"{int(p * 0.5 % 360)}°" for p in tick_positions]

    for momentum, cluster_concentration in results_per_momentum.items():
        df = pd.DataFrame(cluster_concentration)
        x = np.arange(len(df))
        color = colors.get(momentum, "gray")
        label = f"m={momentum}"

        axes[0].plot(x, df["A_percent_in_A_cluster"], label=label, color=color, linewidth=2)
        axes[1].plot(x, df["B_percent_in_B_cluster"], label=label, color=color, linewidth=2)

    for ax, title in zip(
        axes,
        [
            "% A predicted correctly (base cluster)",
            "% B predicted correctly (opposite cluster)",
        ],
    ):
        ax.set_ylim(0, 100)
        ax.set_xlabel("Rotation angle")
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.4)
        ax.legend()

    plt.suptitle(
        "Effect of Momentum Encoder on Rotation Tracking\n"
        "(m=0: pure self-training, m→1: strong temporal memory)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MOMENTUM_VALUES = [0, 0.9, 0.99, 0.999]
    NUM_ITERATIONS = 720  # 360° / 0.5° per step = full rotation

    all_results = {}

    for m in MOMENTUM_VALUES:
        print(f"\n{'='*60}")
        print(f"Running experiment: momentum = {m}")
        print(f"{'='*60}\n")
        cluster_concentration = run_experiment(momentum_value=m, num_iterations=NUM_ITERATIONS)
        all_results[m] = cluster_concentration

    plot_momentum_comparison(all_results, save_path="momentum_comparison.png")
    print("\nAll experiments done.")
