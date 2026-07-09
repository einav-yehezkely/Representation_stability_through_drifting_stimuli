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
from shufflenet_v2_x0_5_MSE_last_epoch import train_model, get_dataloaders, create_model_and_optim, get_dataloaders_from_lists, train_model_fast_for_self_training
from matplotlib.patches import Circle
import time
import torch.optim as optim
from torch.optim import lr_scheduler

BASE_DIR = "tmp_1epoch_0.1degs"
os.makedirs(BASE_DIR, exist_ok=True)

OUTPUT_DIR = "output_1epoch_0.1degs"
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


SCATTER_DIR = inside_output("scatter_frames")
LINEAR_DIR = inside_output("linear_frames")

os.makedirs(SCATTER_DIR, exist_ok=True)
os.makedirs(LINEAR_DIR, exist_ok=True)


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
    Not anymore: Find the k nearest images to center_point and copy them to output_dir.
    Do: Find the k nearest images to center_point and save their filenames.
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
    # if os.path.exists(output_dir):
    #     # Iterate through all files and folders in the output directory
    #     for filename in os.listdir(output_dir):
    #         file_path = os.path.join(output_dir, filename)
    #         try:
    #             # If it's a file or symbolic link, delete it
    #             if os.path.isfile(file_path) or os.path.islink(file_path):
    #                 os.unlink(file_path)
    #             # If it's a directory, delete it and all its contents
    #             elif os.path.isdir(file_path):
    #                 shutil.rmtree(file_path)
    #         except Exception as e:
    #             # If something goes wrong, print a warning message
    #             print(f"Failed to delete {file_path}. Reason: {e}")
    # else:
    #     # If the directory does not exist, create it
    #     os.makedirs(output_dir)

    # Compute distances
    dists = np.linalg.norm(all_points - center_point, axis=1)
    # nearest_indices = np.argsort(dists)[:k]

    # Use np.argpartition for efficiency, then sort the selected indices
    nearest_indices = np.argpartition(dists, k)[:k]
    nearest_indices = nearest_indices[np.argsort(dists[nearest_indices])]

    selected_names = []

    for idx in nearest_indices:
        name = all_names[idx]
        selected_names.append(name)
        # src_path = os.path.join(image_source_dir, name)
        # dst_path = os.path.join(output_dir, name)
        # selected_names.append(name)
        # try:
        #     shutil.copy2(src_path, dst_path)
        # except FileNotFoundError:
        #     print(f"Warning: {src_path} not found.")

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
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def safe_read_filenames(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return pd.Series(dtype="object")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.Series(dtype="object")

    if "filename" not in df.columns:
        return pd.Series(dtype="object")

    return df["filename"]


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

        # with torch.no_grad():
        #     output = model(input_tensor)
        #     # prediction is the index with max probability
        #     pred = output.argmax(dim=1).item()
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


def classify_images_batched(model, csv_path, clusters=False, batch_size=50):
    model.eval()
    df = pd.read_csv(csv_path)

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

    batch_tensors = []
    batch_rows = []

    with torch.no_grad():
        for _, row in df.iterrows():
            image_path = os.path.join("female_faces", row["filename"])

            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found. Skipping.")
                continue

            image = Image.open(image_path).convert("RGB")
            x = transform(image)

            batch_tensors.append(x)
            batch_rows.append(row)

            if len(batch_tensors) == batch_size:
                process_classification_batch(
                    model, batch_tensors, batch_rows,
                    predicted_A, predicted_B, training_records, clusters
                )
                batch_tensors = []
                batch_rows = []

        if len(batch_tensors) > 0:
            process_classification_batch(
                model, batch_tensors, batch_rows,
                predicted_A, predicted_B, training_records, clusters
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
    else:
        df_A.to_csv(inside_tmp("predicted_as_A.csv"), index=False)
        df_B.to_csv(inside_tmp("predicted_as_B.csv"), index=False)


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
            training_records.append(
                {
                    "filename": row["filename"],
                    "prob_A": prob_a,
                    "pred": "A" if pred == 0 else "B",
                }
            )

def split_and_copy_images(
    csv_path,
    label,
    image_source_dir="female_faces",
    train_ratio=0.8,
    root_dir=inside_tmp("split_data"),
):
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


def create_prediction_scatter(angle, frame_id, save_dir=SCATTER_DIR):
    """
    Create a scatter plot showing model predictions over 2D PCA space.
    Saves the result as an image in the specified folder (default: 'frames').

    Args:
        frame_id (int): Frame number for the filename.
        save_dir (str): Directory to save the image.
    """
    opposite_angle = (angle + 180) % 360
    os.makedirs(save_dir, exist_ok=True)

    # Load PCA data
    df = pd.read_csv("pca_top2_filtered_female.csv", header=None)
    df.columns = ["name", "x", "y"]

    # Load predictions
    pred_a = safe_read_filenames(inside_tmp("predicted_as_A.csv"))
    pred_b = safe_read_filenames(inside_tmp("predicted_as_B.csv"))

    predicted_cluster_a = safe_read_filenames(inside_tmp("cluster_predicted_as_A.csv"))
    predicted_cluster_b = safe_read_filenames(inside_tmp("cluster_predicted_as_B.csv"))

    df_a = df[df["name"].isin(pred_a)]
    df_b = df[df["name"].isin(pred_b)]
    df_predicted_cluster_a = df[df["name"].isin(predicted_cluster_a)]
    df_predicted_cluster_b = df[df["name"].isin(predicted_cluster_b)]

    plt.figure(figsize=(10, 10))
    plt.scatter(df["x"], df["y"], s=5, alpha=0.3, color="gray", label="All Vectors")
    plt.scatter(
        df_predicted_cluster_a["x"],
        df_predicted_cluster_a["y"],
        s=9,
        alpha=0.8,
        color="lightblue",
        label="Trained A - predicted",
    )
    plt.scatter(
        df_predicted_cluster_b["x"],
        df_predicted_cluster_b["y"],
        s=9,
        alpha=0.7,
        color="pink",
        label="Trained B - predicted",
    )
    plt.scatter(
        df_a["x"], df_a["y"], s=10, alpha=0.7, color="blue", label="Predicted A"
    )
    plt.scatter(df_b["x"], df_b["y"], s=10, alpha=0.7, color="red", label="Predicted B")

    # Add circle and lines for reference
    radius = max(np.sqrt(df["x"] ** 2 + df["y"] ** 2)) * 1.05
    circle = Circle(
        (0, 0), radius, fill=False, color="black", linestyle="--", alpha=0.5
    )
    plt.gca().add_patch(circle)
    for angle_circ in range(0, 360, 20):
        rad = np.deg2rad(angle_circ)
        x = radius * np.cos(rad)
        y = radius * np.sin(rad)
        plt.plot([0, x], [0, y], color="gray", linewidth=0.5, alpha=0.5)
        plt.text(x * 1.05, y * 1.05, f"{angle_circ}°", ha="center", va="center")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title(
        f"images predicted as A/B, trained on {angle:.1f}° and {opposite_angle:.1f}° clusters"
    )
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"scatter_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    print(f"\nScatter Frame {frame_id} completed.\n")
    plt.close()


def create_linear_graph(angle, frame_id, save_dir=LINEAR_DIR):
    opposite_angle = (angle + 180) % 360
    os.makedirs(save_dir, exist_ok=True)

    pred_a = pd.DataFrame(
        {"filename": safe_read_filenames(inside_tmp("predicted_as_A.csv"))}
    )
    pred_b = pd.DataFrame(
        {"filename": safe_read_filenames(inside_tmp("predicted_as_B.csv"))}
    )

    pred_a["pred"] = "A"
    pred_b["pred"] = "B"

    df = pd.concat([pred_a, pred_b], ignore_index=True)

    # use precomputed PCA angles
    df["angle_deg"] = df["filename"].map(ANGLE_MAP)
    df = df.dropna(subset=["angle_deg"])

    window_size = 20
    results = []

    for step_angle in range(0, 360, 1):
        end = (step_angle + window_size) % 360

        if step_angle < end:
            window_data = df[(df["angle_deg"] >= step_angle) & (df["angle_deg"] < end)]
        else:
            window_data = df[(df["angle_deg"] >= step_angle) | (df["angle_deg"] < end)]

        total = len(window_data)

        if total > 0:
            count_a = (window_data["pred"] == "A").sum()
            percent_a = count_a * 100 / total
            percent_b = 100 - percent_a
        else:
            percent_a = 0
            percent_b = 0

        center_angle = (step_angle + window_size / 2) % 360
        results.append((center_angle, percent_a, percent_b))

    df_results = pd.DataFrame(results, columns=["angle", "percent_A", "percent_B"])

    angle0 = df_results[df_results["angle"] == 0]
    angle360 = angle0.copy()
    angle360["angle"] = 360
    df_results = pd.concat([df_results, angle360], ignore_index=True)
    df_results = df_results.sort_values(by="angle")

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_results["angle"], df_results["percent_A"], label="Predicted A", color="blue"
    )
    plt.plot(
        df_results["angle"], df_results["percent_B"], label="Predicted B", color="red"
    )

    plt.xlabel("Angle")
    plt.ylabel("%")
    plt.title(
        f"% images predicted as A/B, trained on {angle:.1f}° and {opposite_angle:.1f}° clusters, {window_size}° slices"
    )

    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.axvline(x=angle, color="blue", linewidth=1, linestyle="--")
    plt.axvline(x=opposite_angle, color="red", linewidth=1, linestyle="--")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 100)
    plt.xlim(0, 360)

    path = os.path.join(save_dir, f"linear_frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=300)
    print(f"\nLinear Frame {frame_id} completed.\n")
    plt.close()


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

        # with torch.no_grad():
        #     pred = model(x).argmax(dim=1).item()
        with torch.no_grad():
            output = model(x)
            prob_b = torch.sigmoid(output).item()
            pred = 1 if prob_b >= 0.5 else 0

        count_target += int(pred == target_pred)
        total += 1

    percent = (100 * count_target / total) if total > 0 else 0
    return percent, total

def percent_predicted_as_filenames_batched(
    model,
    filenames,
    target_pred,
    image_source_dir="female_faces",
    batch_size=50,
):
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
    batch_tensors = []

    with torch.no_grad():
        for fname in filenames:
            img_path = os.path.join(image_source_dir, fname)
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert("RGB")
            x = transform(img)
            batch_tensors.append(x)

            if len(batch_tensors) == batch_size:
                batch = torch.stack(batch_tensors).to(device)

                outputs = model(batch).squeeze(1)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long()

                count_target += (preds == target_pred).sum().item()
                total += len(batch_tensors)

                batch_tensors = []

        if len(batch_tensors) > 0:
            batch = torch.stack(batch_tensors).to(device)

            outputs = model(batch).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            count_target += (preds == target_pred).sum().item()
            total += len(batch_tensors)

    percent = (100 * count_target / total) if total > 0 else 0
    return percent, total

def compute_cluster_concentration(
    angle, iteration, cluster_concentration=None, k_eval=100
):
    """
    Evaluate the current trained classifier after clustering-based training.

    Measures:
      - Among the k_eval images closest to the current A angle,
        what percentage is classified as A by the final sigmoid classifier.
      - Among the k_eval images closest to the opposite B angle,
        what percentage is classified as B by the final sigmoid classifier.

    Note:
    The model was trained using clustering-based pseudo-labels,
    but this function evaluates the final binary classifier output.
    """
    if cluster_concentration is None:
        cluster_concentration = []

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

    # pct_A_in_A, nA = percent_predicted_as_filenames(
    #     self_training_model, eval_A_filenames, target_pred=0
    # )
    # pct_B_in_B, nB = percent_predicted_as_filenames(
    #     self_training_model, eval_B_filenames, target_pred=1
    # )

    pct_A_in_A, nA = percent_predicted_as_filenames_batched(
        self_training_model, eval_A_filenames, target_pred=0
    )

    pct_B_in_B, nB = percent_predicted_as_filenames_batched(
        self_training_model, eval_B_filenames, target_pred=1
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


def compute_angle_concentration_from_csv(
    angle, iteration, window_size=20, sequence_concentration=None
):
    """
    Compute percentage of images predicted as A/B around their respective training angles,

    Parameters
    ----------
    angle : float
        Training angle in degrees (0-360).
    iteration : int
        The current rotation iteration index.
    window_size : float, optional
        Angular window size in degrees (default = 20°, i.e., ±10° around the center).
    sequence_concentration : list, optional
        List to which results will be appended (creates a new one if None).

    Returns
    -------
    sequence_concentration : list of dicts
        Updated list containing concentration data for this iteration.
    """

    if sequence_concentration is None:
        sequence_concentration = []

    # --- Load prediction results ---
    pred_a_path = inside_tmp("predicted_as_A.csv")
    pred_b_path = inside_tmp("predicted_as_B.csv")
    if not (os.path.exists(pred_a_path) and os.path.exists(pred_b_path)):
        print("Warning: prediction CSVs not found, skipping concentration measurement.")
        return sequence_concentration

    pred_a = pd.read_csv(pred_a_path)
    pred_b = pd.read_csv(pred_b_path)

    pred_a["pred"] = "A"
    pred_b["pred"] = "B"

    df = pd.concat([pred_a, pred_b], ignore_index=True)

    # --- Load PCA coordinates for all images ---
    df_full = pd.read_csv("pca_top2_filtered_female.csv", header=None)
    df_full.columns = ["filename", "x", "y"]

    df = df.merge(df_full, on="filename", how="left")

    # Compute angle of each image in PCA space
    df["angle_deg"] = np.degrees(np.arctan2(df["y"], df["x"])) % 360

    def select_window(df, center, width):
        """Select subset of df within ±width/2 around center angle."""
        start = (center - width / 2) % 360
        end = (center + width / 2) % 360
        if start < end:
            return df[(df["angle_deg"] >= start) & (df["angle_deg"] < end)]
        else:
            return df[(df["angle_deg"] >= start) | (df["angle_deg"] < end)]

    # --- Compute A/B concentration around respective angles ---
    opposite_angle = (angle + 180) % 360
    window_train = select_window(df, angle, window_size)
    window_opposite = select_window(df, opposite_angle, window_size)

    percent_A = 0
    percent_B = 0
    if len(window_train) > 0:
        percent_A = (window_train["pred"] == "A").sum() * 100 / len(window_train)
    if len(window_opposite) > 0:
        percent_B = (window_opposite["pred"] == "B").sum() * 100 / len(window_opposite)

    sequence_concentration.append(
        {
            "iteration": iteration,
            "angle": angle,
            "A_percent_near_train_angle": percent_A,
            "B_percent_near_opposite_angle": percent_B,
        }
    )

    print(
        f"Iteration {iteration}: {percent_A:.1f}% A near {angle:.1f}°, {percent_B:.1f}% B near {opposite_angle:.1f}°\n"
    )

    return sequence_concentration


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
        f"Are Classified as the Cluster’s Intended Label\n"
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


if __name__ == "__main__":

    UNSUPERVISED = True
    ROTATION_DEGS = 0.1
    NUM_ITERATIONS = 1
    NUM_EPOCHS = 1
    PLOT_EVERY = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    names, points = load_top2_filtered("pca_top2_filtered_female.csv")
    base_point, opposite_point = create_base_and_opposite_points(0)
    self_training_model = load_model(model_path="model_ft_0_MSE.pth")
    self_training_model = self_training_model.to(device)
    optimizer_ft = optim.Adam(
        self_training_model.parameters(),
        lr=0.0001,
    )
    # Generate rotation sequence
    rotation_seq, _ = generate_rotation_sequence(
        base_point=base_point,
        all_points=points,
        all_names=names,
        num_steps=360,
        start_angle=0,
        rotation_range=360,
        used_indices=set(),  # ensure we don't reuse the same images to train on in the rotation sequence, so we get 360 unique images in the sequence (one per degree)
    )

    df_seq = pd.DataFrame(rotation_seq, columns=["step", "angle_deg", "filename"])
    df_seq.to_csv(inside_tmp("rotation_sequence_all.csv"), index=False)
    print("Generated rotation sequence with unique images at each step.\n")

    cluster_concentration = []
    training_log = []
    # classification_error_log = []

    start = time.time()
    for i in range(
        NUM_ITERATIONS
    ): 
        collect_nearest_images(
            base_point, points, names, output_dir=inside_tmp("A"), k=200
        )
        collect_nearest_images(
            opposite_point, points, names, output_dir=inside_tmp("B"), k=200
        )
        # now we have two directories: A and B with 200 images each from opposite clusters
        if UNSUPERVISED:
            merge_clusters()
            t = time.time()
            training_records = classify_images_batched(
                self_training_model,
                inside_tmp("filenames_merged.csv"),
                clusters=True,
                batch_size=50,
            )

            print("Classified images in clusters A and B.")
            print(f"classify_images time: {time.time() - t:.2f}s")
            # angle_deg = np.degrees(np.arctan2(base_point[1], base_point[0])) % 360
            angle_deg = (i * ROTATION_DEGS) % 360

            for rec in training_records:
                image_angle = ANGLE_MAP.get(rec["filename"], np.nan)

                training_log.append(
                    {
                        "iteration": i,
                        "cluster_angle": angle_deg,
                        "filename": rec["filename"],
                        "image_angle": image_angle,
                        "cluster_label": rec["pred"],
                        "prob_A": rec["prob_A"],
                    }
                )

            # error_info = compute_cluster_classification_errors(
            #     self_training_model,
            #     iteration=i,
            #     angle=angle_deg,
            # )

            # classification_error_log.append(error_info)
            

            # print(
            #     f"Iteration {i}: "
            #     f"A wrong = {error_info['wrong_A_count']}/200 | "
            #     f"B wrong = {error_info['wrong_B_count']}/200"
            # )

            # now there are two CSVs: cluster_predicted_as_A.csv and cluster_predicted_as_B.csv
            # split_and_copy_images(inside_tmp("cluster_predicted_as_A.csv"), label="A")
            # split_and_copy_images(inside_tmp("cluster_predicted_as_B.csv"), label="B")
            # now we have a split_data/train/A and split_data/val/A

            df_A = pd.read_csv(inside_tmp("cluster_predicted_as_A.csv"))
            df_B = pd.read_csv(inside_tmp("cluster_predicted_as_B.csv"))
        else:
            # split_and_copy_images(inside_tmp("filenames_A.csv"), label="A")
            # split_and_copy_images(inside_tmp("filenames_B.csv"), label="B")

            df_A = pd.read_csv(inside_tmp("filenames_A.csv"))
            df_B = pd.read_csv(inside_tmp("filenames_B.csv"))
            # now we have a split_data/train/A and split_data/val/A

        # dataloaders, dataset_sizes, class_names = get_dataloaders(
        #     data_dir=inside_tmp("split_data")
        # )
        filenames = df_A["filename"].tolist() + df_B["filename"].tolist()
        labels = [0] * len(df_A) + [1] * len(df_B)

        dataloaders, dataset_sizes, class_names = get_dataloaders_from_lists(
            filenames,
            labels,
            image_dir="female_faces",
            batch_size=100,
        )
        # _, criterion, optimizer_ft, exp_lr_scheduler = create_model_and_optim()
        ################
        criterion = nn.MSELoss()  
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=5, gamma=1
        )  # gamma=0.1, right now no LR decay
        ################
        if UNSUPERVISED == True:
            self_training_model = train_model_fast_for_self_training(
                self_training_model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                num_epochs=NUM_EPOCHS,
            )
        else:
            self_training_model = train_model(
                self_training_model,
                dataloaders,
                dataset_sizes,
                criterion,
                optimizer_ft,
                exp_lr_scheduler,
                num_epochs=NUM_EPOCHS,
                plots = False
            )

        # # now we have a trained model - self trained on it's own predictions


        # classify_images(
        #     self_training_model,
        #     csv_path=inside_tmp("rotation_sequence_all.csv"),
        #     clusters=False,
        # )
        # print("Classified rotation sequence.")

        # # now we have two CSVs: predicted_as_A.csv and predicted_as_B.csv
        # # create scatter plot of predictions
        # if UNSUPERVISED:
        #     create_prediction_scatter(angle=angle_deg, frame_id=i)
        # # create linear graph of predictions
        # create_linear_graph(angle=angle_deg, frame_id=i)

        # Plotting and evaluation every PLOT_EVERY iterations
        if i % PLOT_EVERY == 0:
            # classify_images(
            #     self_training_model,
            #     csv_path=inside_tmp("rotation_sequence_all.csv"),
            #     clusters=False,
            # )
            t = time.time()

            classify_images_batched(
                self_training_model,
                csv_path=inside_tmp("rotation_sequence_all.csv"),
                clusters=False,
                batch_size=50,
            )

            print("Classified rotation sequence.")
            print(f"rotation sequence classification time: {time.time()-t:.2f}s")
            

            if UNSUPERVISED:
                create_prediction_scatter(angle=angle_deg, frame_id=i)

            create_linear_graph(angle=angle_deg, frame_id=i)


        # compute concentration of predictions around training and opposite angles
        # cluster_concentration = compute_cluster_concentration(
        #     angle=angle_deg,
        #     iteration=i,
        #     cluster_concentration=cluster_concentration,
        # )
        t = time.time()

        cluster_concentration = compute_cluster_concentration(
            angle=angle_deg,
            iteration=i,
            cluster_concentration=cluster_concentration,
        )

        print(f"Concentration time: {time.time()-t:.2f}s")
        # create a scatter plot of the predictions
        # save the scatter plot in the frames directory
        # rotate base_point and opposite_point by 5 degrees for the next iteration
        base_point = rotate_vector(base_point, angle_deg=ROTATION_DEGS)  
        opposite_point = rotate_vector(opposite_point, angle_deg=ROTATION_DEGS) 
        # clean up A and B directories for the next iteration
        # shutil.rmtree(inside_tmp("A"), ignore_errors=True)
        # shutil.rmtree(inside_tmp("B"), ignore_errors=True)
        # shutil.rmtree(inside_tmp("split_data"), ignore_errors=True)
        # clean up the split_data directory for the next iteration
        # delete csv files
        csv_files_to_delete = [
            inside_tmp("filenames_A.csv"),
            inside_tmp("filenames_B.csv"),
            inside_tmp("predicted_as_A.csv"),
            inside_tmp("predicted_as_B.csv"),
        ]

        if UNSUPERVISED:
            csv_files_to_delete += [
                inside_tmp("filenames_merged.csv"),
                inside_tmp("cluster_predicted_as_A.csv"),
                inside_tmp("cluster_predicted_as_B.csv"),
            ]

        for fname in csv_files_to_delete:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except Exception as e:
                    print(f"Could not delete {fname}: {e}")
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    print("Total time (minutes): ", (end - start) / 60, "minutes")
    plot_cluster_concentration(
        cluster_concentration,
        save_path=inside_output("cluster_concentration_over_rotations.png"),
        type_of_learning=(
            "Unsupervised Learning" if UNSUPERVISED else "Supervised Learning"
        ),
    )

    pd.DataFrame(training_log).to_csv(inside_output("training_log.csv"), index=False)

    changed_csv, summary = save_label_change_csvs(inside_output("training_log.csv"))

    torch.save(
        self_training_model.state_dict(), inside_output("model_self_trained.pth")
    )

    # df_errors = pd.DataFrame(classification_error_log)

    # df_errors.to_csv(inside_output("cluster_classification_errors.csv"), index=False)

    # plt.figure(figsize=(12, 6))

    # plt.plot(
    #     df_errors["iteration"],
    #     df_errors["wrong_A_count"],
    #     marker="o",
    #     label="Wrong in cluster A",
    # )

    # plt.plot(
    #     df_errors["iteration"],
    #     df_errors["wrong_B_count"],
    #     marker="o",
    #     label="Wrong in cluster B",
    # )

    # plt.xlabel("Iteration")
    # plt.ylabel("Number of incorrectly classified images")

    # plt.title(
    #     "Wrongly Classified Images in the 200 Closest Images\n" "Before Self-Training"
    # )

    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()

    # plt.savefig(inside_output("cluster_classification_errors.png"), dpi=300)

    # plt.close()
