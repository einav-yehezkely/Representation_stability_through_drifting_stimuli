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
from shufflenet_v2_x0_5 import train_model, get_dataloaders, create_model_and_optim
from matplotlib.patches import Circle
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import random


BASE_DIR = "tmp"
os.makedirs(BASE_DIR, exist_ok=True)


def inside_tmp(*paths):
    """Return a path inside the BASE_DIR (tmp)."""
    return os.path.join(BASE_DIR, *paths)


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


def merge_sequences(
    path_a=inside_tmp("rotation_sequence_A.csv"),
    path_b=inside_tmp("rotation_sequence_B.csv"),
    to_csv=inside_tmp("merged_sequences.csv"),
):
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    i, j = 0, 0
    rows = []

    while i < len(df_a) or j < len(df_b):
        if i == len(df_a):
            row = df_b.iloc[j].copy()
            row["group"] = "B"
            rows.append(row)
            j += 1
        elif j == len(df_b):
            row = df_a.iloc[i].copy()
            row["group"] = "A"
            rows.append(row)
            i += 1
        else:
            if random.random() < 0.5:
                row = df_a.iloc[i].copy()
                row["group"] = "A"
                rows.append(row)
                i += 1
            else:
                row = df_b.iloc[j].copy()
                row["group"] = "B"
                rows.append(row)
                j += 1

    # Convert list of Series to DataFrame
    merged_df = pd.DataFrame(rows)

    # Move 'group' column to the front
    cols = ["group"] + [col for col in merged_df.columns if col != "group"]
    merged_df = merged_df[cols]

    # Save to CSV
    merged_df.to_csv(to_csv, index=False)


def load_model(model_path="model_ft_no_reg_0.pth"):
    """
    Load the pre-trained model for classification.
    The model is trained on images from 135 and 315 degrees.
    """
    model = models.shufflenet_v2_x0_5(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def classify_images(model, csv_path, clusters=False):
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

    for i, row in df.iterrows():
        image_path = os.path.join("female_faces", row["filename"])

        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found. Skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            # prediction is the index with max probability
            pred = output.argmax(dim=1).item()

        # Track predictions
        if pred == 0:
            predicted_A.append(row)
        else:
            predicted_B.append(row)

    # Save predicted CSVs
    if clusters:
        pd.DataFrame(predicted_A).to_csv(
            inside_tmp("cluster_predicted_as_A.csv"), index=False
        )
        pd.DataFrame(predicted_B).to_csv(
            inside_tmp("cluster_predicted_as_B.csv"), index=False
        )
    else:
        pd.DataFrame(predicted_A).to_csv(inside_tmp("predicted_as_A.csv"), index=False)
        pd.DataFrame(predicted_B).to_csv(inside_tmp("predicted_as_B.csv"), index=False)


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


def create_prediction_scatter(angle, frame_id, save_dir="scatter_frames"):
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
    pred_a = pd.read_csv(inside_tmp("predicted_as_A.csv"))["filename"]
    pred_b = pd.read_csv(inside_tmp("predicted_as_B.csv"))["filename"]

    predicted_cluster_a = pd.read_csv(
        inside_tmp("cluster_predicted_as_A.csv"), header=None
    )[0]
    predicted_cluster_b = pd.read_csv(
        inside_tmp("cluster_predicted_as_B.csv"), header=None
    )[0]

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
    print(f"\nScatter Frame {frame_id} completed. Model re-trained and evaluated.\n")
    plt.close()


def create_linear_graph(angle, frame_id, save_dir="linear_frames"):
    opposite_angle = (angle + 180) % 360
    os.makedirs(save_dir, exist_ok=True)

    pred_a = pd.read_csv(inside_tmp("predicted_as_A.csv"))
    pred_b = pd.read_csv(inside_tmp("predicted_as_B.csv"))

    pred_a["pred"] = "A"
    pred_b["pred"] = "B"

    df = pd.concat([pred_a, pred_b], ignore_index=True)

    df_full = pd.read_csv("pca_top2_filtered_female.csv", header=None)
    df_full.columns = ["filename", "x", "y"]

    df = df.merge(df_full, on="filename", how="left")

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
            count_b = (window_data["pred"] == "B").sum()
            percent_a = count_a * 100 / total
            percent_b = count_b * 100 / total
        else:
            percent_a = 0
            percent_b = 0

        # Use center of window for plotting
        center_angle = (step_angle + window_size / 2) % 360
        results.append((center_angle, percent_a, percent_b))

    df_results = pd.DataFrame(results, columns=["angle", "percent_A", "percent_B"])

    # Add closing point at 360°
    angle0 = df_results[df_results["angle"] == 0]
    # 360° is the same as 0°
    angle360 = angle0.copy()
    angle360["angle"] = 360
    df_results = pd.concat([df_results, angle360], ignore_index=True)
    # sort by angle for proper plotting
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
    print(f"\nLinear Frame {frame_id} completed. Model re-trained and evaluated.\n")
    plt.close()


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


def plot_angle_concentration(
    sequence_concentration,
    window_size=20,
    save_path="linear_concentration_over_rotations.png",
):
    """
    Plot concentration of predictions around training and opposite angles over all iterations.
    """
    df_seq = pd.DataFrame(sequence_concentration)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_seq["iteration"],
        df_seq["A_percent_near_train_angle"],
        label="A near training angle",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        df_seq["iteration"],
        df_seq["B_percent_near_opposite_angle"],
        label="B near opposite angle",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Rotation iteration (0-71)")
    plt.ylabel(f"% predicted correctly within ±{window_size/2}° window")
    plt.title(
        f"Concentration of correct predictions around training and opposite angles ({window_size}° slices)\n--Supervised Learning--"
    )
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


names, points = load_top2_filtered("pca_top2_filtered_female.csv")
base_point, opposite_point = create_base_and_opposite_points(0)
self_training_model = load_model(model_path="model_ft_no_reg_0.pth")

if __name__ == "__main__":
    sequence_concentration = []
    start = time.time()
    for i in range(72):  # 360/5=72
        collect_nearest_images(
            base_point, points, names, output_dir=inside_tmp("A"), k=1000
        )
        collect_nearest_images(
            opposite_point, points, names, output_dir=inside_tmp("B"), k=1000
        )
        # now we have two directories: A and B with 1000 images each from opposite clusters
        ######################### unsupervised learning
        # we can now classify these images using the model
        merge_clusters()
        # now we have a merged CSV with filenames from both clusters
        classify_images(
            self_training_model,
            csv_path=inside_tmp("filenames_merged.csv"),
            clusters=True,
        )  # classify clusters A and B
        """
        print("Classified images in clusters A and B.")
        # now there are two CSVs: cluster_predicted_as_A.csv and cluster_predicted_as_B.csv
        ### we will now retrain the model on these classifications ###
        split_and_copy_images(csv_path="cluster_predicted_as_A.csv", label="A")
        # now we have a split_data/train/A and split_data/val/A
        split_and_copy_images(csv_path="cluster_predicted_as_B.csv", label="B")
        """
        #########################
        ######################### supervised learning
        split_and_copy_images(csv_path=inside_tmp("filenames_A.csv"), label="A")
        # now we have a split_data/train/A and split_data/val/A
        split_and_copy_images(csv_path=inside_tmp("filenames_B.csv"), label="B")
        # to use unsupervised learning, comment out the above two lines and uncomment the previous two lines
        #########################
        # now we have a split_data/train/B and split_data/val/B
        dataloaders, dataset_sizes, class_names = get_dataloaders(
            data_dir=inside_tmp("split_data")
        )
        # _, criterion, optimizer_ft, exp_lr_scheduler = create_model_and_optim()
        ################
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.AdamW(
            self_training_model.parameters(), lr=0.005, weight_decay=0.01
        )
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
        ################
        self_training_model = train_model(
            self_training_model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer_ft,
            exp_lr_scheduler,
            num_epochs=4,
            plots=False,
        )
        # Generate rotation sequences
        used = set()

        rotation_seq_A, used = generate_rotation_sequence(
            base_point=base_point,
            all_points=points,
            all_names=names,
            num_steps=180,
            start_angle=0,
            rotation_range=180,
            used_indices=used,
        )
        df_A = pd.DataFrame(rotation_seq_A, columns=["step", "angle_deg", "filename"])
        df_A.to_csv(inside_tmp("rotation_sequence_A.csv"), index=False)
        print("Saved rotation sequence A to rotation_sequence_A.csv")

        rotation_seq_B, used = generate_rotation_sequence(
            base_point=opposite_point,
            all_points=points,
            all_names=names,
            num_steps=180,
            start_angle=0,
            rotation_range=180,
            used_indices=used,
        )
        df_B = pd.DataFrame(rotation_seq_B, columns=["step", "angle_deg", "filename"])
        df_B.to_csv(inside_tmp("rotation_sequence_B.csv"), index=False)
        print("Saved rotation sequence B to rotation_sequence_B.csv")

        merge_sequences()  # merge the two sequences into one CSV
        # now we have a trained model - self trained on it's own predictions
        classify_images(
            self_training_model,
            csv_path=inside_tmp("merged_sequences.csv"),
            clusters=False,
        )  # classify rotation sequence
        print("Classified rotation sequence.")

        angle_deg = np.degrees(np.arctan2(base_point[1], base_point[0])) % 360
        # now we have two CSVs: predicted_as_A.csv and predicted_as_B.csv
        # create scatter plot of predictions
        create_prediction_scatter(angle=angle_deg, frame_id=i)
        # create linear graph of predictions
        create_linear_graph(angle=angle_deg, frame_id=i)
        # compute concentration of predictions around training and opposite angles
        sequence_concentration = compute_angle_concentration_from_csv(
            angle=angle_deg,
            iteration=i,
            window_size=20,
            sequence_concentration=sequence_concentration,
        )
        # create a scatter plot of the predictions
        # save the scatter plot in the frames directory
        # rotate base_point and opposite_point by 5 degrees for the next iteration
        base_point = rotate_vector(base_point, angle_deg=5)
        opposite_point = rotate_vector(opposite_point, angle_deg=5)
        # clean up A and B directories for the next iteration
        shutil.rmtree(inside_tmp("A"), ignore_errors=True)
        shutil.rmtree(inside_tmp("B"), ignore_errors=True)
        shutil.rmtree(inside_tmp("split_data"), ignore_errors=True)
        # clean up the split_data directory for the next iteration
        # delete csv files
        csv_files_to_delete = [
            inside_tmp("filenames_A.csv"),
            inside_tmp("filenames_B.csv"),
            inside_tmp("filenames_merged.csv"),
            inside_tmp("predicted_as_A.csv"),
            inside_tmp("predicted_as_B.csv"),
            inside_tmp("rotation_sequence_A.csv"),
            inside_tmp("rotation_sequence_B.csv"),
            inside_tmp("merged_sequences.csv"),
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
    plot_angle_concentration(sequence_concentration, window_size=20)
    torch.save(self_training_model.state_dict(), "model_self_trained.pth")
