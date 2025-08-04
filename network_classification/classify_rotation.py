import os
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from operator import ne
import numpy as np
import pandas as pd
import shutil


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


names, points = load_top2_filtered("pca_top2_filtered_female.csv")
base_point, opposite_point = create_base_and_opposite_points(135)


for i in range(72):  # 360/5=72
    rotated_base = rotate_vector(base_point, angle=5)
    rotated_opposite = rotate_vector(opposite_point, angle=5)
    base_indices = collect_nearest_images(
        rotated_base,
        points,
        names,
        output_dir="A",
    )
    opp_indices = collect_nearest_images(
        rotated_opposite,
        points,
        names,
        output_dir="B",
    )
    model = load_model(model_path="model_ft_no_reg_A_vs_B_135.pth")
    merge_clusters()
    # now we have a merged CSV with filenames from both clusters
    classify_images(model, csv_path="filenames_merged.csv")  # clusters
    # now there are two CSVs: predicted_as_A.csv and predicted_as_B.csv
