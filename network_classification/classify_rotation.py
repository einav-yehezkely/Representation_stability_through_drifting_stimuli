import os
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ResNet18 import train_model, get_dataloaders, create_model_and_optim
from merge_sequences import merge_sequences


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
    k=200,
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
    csv_path = f"filenames_{output_dir}.csv"
    pd.DataFrame(selected_names, columns=["filename"]).to_csv(csv_path, index=False)
    print(f"Saved {len(selected_names)} image names to {csv_path}")

    return nearest_indices


def merge_clusters():
    # Load both CSVs
    df_a = pd.read_csv("filenames_A.csv")
    df_b = pd.read_csv("filenames_B.csv")

    # Concatenate the DataFrames
    df_merged = pd.concat([df_a, df_b], ignore_index=True)

    # Save to new CSV
    df_merged.to_csv("filenames_merged.csv", index=False)


def load_model(model_path="model_ft_no_reg_A_vs_B_135.pth"):
    """
    Load the pre-trained model for classification.
    The model is trained on images from 135 and 315 degrees.
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: A and B
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def classify_images(model, csv_path):
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
    pd.DataFrame(predicted_A).to_csv("predicted_as_A.csv", index=False)
    pd.DataFrame(predicted_B).to_csv("predicted_as_B.csv", index=False)


def split_and_copy_images(
    csv_path,
    label,
    image_source_dir="female_faces",
    train_ratio=0.8,
    root_dir="split_data",
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


def generate_rotation_sequence(base_point, all_points, all_names, num_steps=1000):
    """
    Rotate base_point around the origin in num_steps steps (in degrees)
    and find the closest point from all_points at each step.

    Parameters:
    - base_point: 2D numpy array representing the starting point
    - all_points: numpy array of shape (N, 2), 2D positions of all images
    - all_names: list or array of N image names corresponding to all_points
    - num_steps: number of rotation steps (default 1000 = every 0.36 degrees)

    Returns:
    - List of tuples: (step_index, angle_in_degrees, closest_image_name)
    """
    results = []
    used_indices = set()

    for i in range(num_steps):
        angle_deg = 360 * i / num_steps  # degrees
        rotated = rotate_vector(base_point, angle_deg)

        dists = np.linalg.norm(all_points - rotated, axis=1)

        for idx in used_indices:
            dists[idx] = np.inf  # Ignore already used indices

        idx_closest = np.argmin(dists)
        used_indices.add(idx_closest)
        results.append((i, angle_deg, all_names[idx_closest]))
    return results


def create_prediction_scatter(frame_id, save_dir="frames"):
    """
    Create a scatter plot showing model predictions over 2D PCA space.
    Saves the result as an image in the specified folder (default: 'frames').

    Args:
        frame_id (int): Frame number for the filename.
        save_dir (str): Directory to save the image.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load PCA data
    df = pd.read_csv("pca_top2_filtered_female.csv", header=None)
    df.columns = ["name", "x", "y"]

    # Load predictions
    pred_a = pd.read_csv("predicted_as_A.csv", header=None)[3]
    pred_b = pd.read_csv("predicted_as_B.csv", header=None)[3]

    cluster_a = pd.read_csv("filenames_A.csv", header=None)[0]
    cluster_b = pd.read_csv("filenames_B.csv", header=None)[0]

    df_a = df[df["name"].isin(pred_a)]
    df_b = df[df["name"].isin(pred_b)]
    df_cluster_a = df[df["name"].isin(cluster_a)]
    df_cluster_b = df[df["name"].isin(cluster_b)]

    plt.figure(figsize=(10, 10))
    plt.scatter(df["x"], df["y"], s=5, alpha=0.3, color="gray", label="All Vectors")
    plt.scatter(
        df_cluster_a["x"],
        df_cluster_a["y"],
        s=9,
        alpha=0.8,
        color="lightblue",
        label="Trained A",
    )
    plt.scatter(
        df_a["x"], df_a["y"], s=10, alpha=0.7, color="blue", label="Predicted A"
    )
    plt.scatter(
        df_cluster_b["x"],
        df_cluster_b["y"],
        s=9,
        alpha=0.7,
        color="pink",
        label="Trained B",
    )
    plt.scatter(df_b["x"], df_b["y"], s=10, alpha=0.7, color="red", label="Predicted B")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title("Model Predictions")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"frame_{frame_id:03d}.png")
    plt.savefig(path, dpi=150)
    print(f"\nFrame {i} complete. Model re-trained and evaluated.\n")
    plt.close()


names, points = load_top2_filtered("pca_top2_filtered_female.csv")
base_point, opposite_point = create_base_and_opposite_points(135)
self_training_model = load_model(model_path="model_ft_no_reg_A_vs_B_135.pth")

for i in range(72):  # 360/5=72
    base_point = rotate_vector(base_point, angle_deg=5)
    opposite_point = rotate_vector(opposite_point, angle_deg=5)
    collect_nearest_images(
        base_point,
        points,
        names,
        output_dir="A",
    )
    collect_nearest_images(
        opposite_point,
        points,
        names,
        output_dir="B",
    )
    # now we have two directories: A and B with 200 images each from opposite clusters
    # we can now classify these images using the model
    merge_clusters()
    # now we have a merged CSV with filenames from both clusters
    classify_images(
        self_training_model, csv_path="filenames_merged.csv"
    )  # classify clusters A and B
    # now there are two CSVs: predicted_as_A.csv and predicted_as_B.csv
    ### we will now retrain the model on these classifications ###
    split_and_copy_images(csv_path="predicted_as_A.csv", label="A")
    # now we have a split_data/train/A and split_data/val/A
    split_and_copy_images(csv_path="predicted_as_B.csv", label="B")
    # now we have a split_data/train/B and split_data/val/B
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir="split_data")
    _, criterion, optimizer_ft, exp_lr_scheduler = create_model_and_optim()
    self_training_model = train_model(
        self_training_model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=8,
    )
    # Generate rotation sequences
    rotation_seq_A = generate_rotation_sequence(
        base_point=base_point, all_points=points, all_names=names, num_steps=360
    )
    df_A = pd.DataFrame(rotation_seq_A, columns=["step", "angle_deg", "filename"])
    df_A.to_csv("rotation_sequence_A.csv", index=False)
    print("Saved rotation sequence A to rotation_sequence_A.csv")

    rotation_seq_B = generate_rotation_sequence(
        base_point=opposite_point, all_points=points, all_names=names, num_steps=360
    )
    df_B = pd.DataFrame(rotation_seq_B, columns=["step", "angle_deg", "filename"])
    df_B.to_csv("rotation_sequence_B.csv", index=False)
    print("Saved rotation sequence B to rotation_sequence_B.csv")
    merge_sequences()  # merge the two sequences into one CSV
    # now we have a trained model - self trained on it's own predictions
    classify_images(
        self_training_model, csv_path="merged_sequences.csv"
    )  # classify rotation sequence
    # now we have two CSVs: predicted_as_A.csv and predicted_as_B.csv
    create_prediction_scatter(frame_id=i)
    # create a scatter plot of the predictions
    # save the scatter plot in the frames directory
    shutil.rmtree("A", ignore_errors=True)
    shutil.rmtree("B", ignore_errors=True)
    shutil.rmtree("split_data", ignore_errors=True)
    # clean up the split_data directory for the next iteration

torch.save(self_training_model.state_dict(), "model_self_trained.pth")
