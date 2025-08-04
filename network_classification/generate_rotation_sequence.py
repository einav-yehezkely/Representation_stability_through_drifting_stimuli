#################################################################
# generate_rotation_sequence.py

# This module provides tools to generate a sequence of images based on rotation in 2D PCA space.
# The input data is expected to be a filtered subset of face images projected to their top 2 principal components.
# The filtering ensures that selected images are not only close in the 2D space but also have low variance
# in the remaining PCA components — meaning the faces are truly similar in the full embedding space.

# Typical use:
# - Choose a base image in 2D PCA space
# - Rotate its embedding vector around the origin
# - At each step, retrieve the image whose 2D PCA location is closest to the rotated vector
#################################################################


from operator import ne
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def load_top2_filtered(csv_path="pca_top2_filtered_female.csv"):
    """
    Load 2D PCA coordinates of pre-filtered images from a CSV file.

    Assumes the format: image_name, x, y

    The images in this file have been filtered such that they are not only close in the top 2 PCA components,
    but also have low variance in the remaining PCA dimensions (i.e., their radius in the residual components is small).
    This ensures that proximity in 2D reflects real similarity in the full FaceNet space.
    """
    # df = pd.read_csv(csv_path, header=0)
    # names = df["filename"].values
    # x = df["PC1"].values
    # y = df["PC2"].values
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    points = np.stack((x, y), axis=1)
    return names, points


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


def find_opposite(base_point, all_points):
    """
    Find the point in all_points that is closest to the vector pointing in the opposite
    direction of base_point.
    """
    opposite_vec = -base_point
    dists = np.linalg.norm(all_points - opposite_vec, axis=1)
    return np.argmin(dists)


def rotate_and_find_nearest(base_point, all_points, all_names, num_steps=1000):
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


def greedy_walk_nearest_neighbors(base_point, all_points, all_names, num_steps=1000):
    """
    Walk through the 2D space by moving at each step to the closest unused point.

    Parameters:
    - base_point: starting point in 2D space
    - all_points: numpy array of shape (N, 2)
    - all_names: list of image names
    - num_steps: how many steps to take (or less if points run out)

    Returns:
    - List of tuples: (step_index, image_name, current_point)
    """
    results = []
    used_indices = set()

    current_point = base_point

    for i in range(num_steps):
        dists = np.linalg.norm(all_points - current_point, axis=1)

        # Mask already used points
        for idx in used_indices:
            dists[idx] = np.inf

        if np.all(dists == np.inf):
            print("No more unused points available.")
            break

        idx_closest = np.argmin(dists)
        used_indices.add(idx_closest)

        current_point = all_points[idx_closest]
        image_name = all_names[idx_closest]

        results.append((i, image_name, current_point))

    return results


def collect_nearest_images(
    center_point,
    all_points,
    all_names,
    output_dir,
    k=1000,
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
    os.makedirs(output_dir, exist_ok=True)

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
    csv_path = os.path.join(output_dir, "filenames.csv")
    pd.DataFrame(selected_names, columns=["filename"]).to_csv(csv_path, index=False)
    print(f"Saved {len(selected_names)} image names to {csv_path}")

    return nearest_indices


def plot_clusters_with_given_indices(
    base_point,
    opposite_point,
    all_points,
    base_indices,
    opp_indices,
    title="PCA Clusters",
    save_path=None,
):
    plt.figure(figsize=(7, 7))
    plt.scatter(
        all_points[:, 0], all_points[:, 1], color="lightgray", s=5, label="All points"
    )
    plt.scatter(
        all_points[base_indices, 0],
        all_points[base_indices, 1],
        color="blue",
        s=10,
        label="Group A",
    )
    plt.scatter(
        all_points[opp_indices, 0],
        all_points[opp_indices, 1],
        color="red",
        s=10,
        label="Group B",
    )
    plt.scatter(
        base_point[0],
        base_point[1],
        color="black",
        s=40,
        marker="x",
        label="Base point",
    )
    plt.scatter(
        opposite_point[0],
        opposite_point[1],
        color="black",
        s=40,
        marker="*",
        label="Opposite point",
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_two_rotation_paths_fixed_color(
    all_points,
    all_names,
    base_point,
    opposite_point,
    rotation_seq_A,
    rotation_seq_B,
    title="Rotation Sequences (Red & Blue)",
    save_path=None,
):
    """
    Plot PCA space with two fixed-color rotation sequences (A=blue, B=red),
    and black 'X' markers for the base and opposite points.

    Parameters:
        all_points: np.array of shape (N, 2), all PCA points
        all_names: list of all image names, in order
        base_point: np.array of shape (2,)
        opposite_point: np.array of shape (2,)
        rotation_seq_A: list of (step, angle, filename)
        rotation_seq_B: list of (step, angle, filename)
        title: plot title
        save_path: optional path to save the plot
    """
    # Map filenames to indices
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Get indices of rotation sequences
    indices_A = [name_to_idx[name] for _, _, name in rotation_seq_A]
    indices_B = [name_to_idx[name] for _, _, name in rotation_seq_B]

    plt.figure(figsize=(8, 8))

    # Plot all points in light gray
    plt.scatter(
        all_points[:, 0], all_points[:, 1], color="lightgray", s=5, label="All points"
    )

    # Plot rotation sequence A in blue
    plt.scatter(
        all_points[indices_A, 0],
        all_points[indices_A, 1],
        color="blue",
        s=15,
        label="Rotation A",
    )

    # Plot rotation sequence B in red
    plt.scatter(
        all_points[indices_B, 0],
        all_points[indices_B, 1],
        color="red",
        s=15,
        label="Rotation B",
    )

    # Plot base points in black
    plt.scatter(
        base_point[0],
        base_point[1],
        color="black",
        s=60,
        marker="x",
        label="Base point",
    )
    plt.scatter(
        opposite_point[0],
        opposite_point[1],
        color="black",
        s=60,
        marker="x",
        label="Opposite point",
    )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# Load data
names, points = load_top2_filtered("pca_top2_filtered_female.csv")

# Base and opposite points
# base_idx = 0
# base_point = points[base_idx]
# opposite_point = -base_point

# Compute angles (in radians) of each point from the origin
angles = np.arctan2(points[:, 1], points[:, 0])

# Convert angles from radians to degrees, now in range [-180, 180]
angles_deg = np.degrees(angles)
# Shift all angles to be in the range [0, 360)
angles_deg = (angles_deg + 360) % 360

radii = np.linalg.norm(points, axis=1)

# Define the target angle in degrees
target_angle = 90

target_radius = 0.45

angle_error = np.abs(angles_deg - target_angle)
radius_error = np.abs(radii - target_radius)
combined_error = angle_error + radius_error * 100

# Find the index of the point whose angle is closest to the target angle
base_idx = np.argmin(combined_error)
# Retrieve the actual 2D PCA coordinates of the selected base point
base_point = points[base_idx]

opposite_point = -base_point


# save clusters of k=1000 points around base and opposite points
base_indices = collect_nearest_images(base_point, points, names, output_dir="A")
opp_indices = collect_nearest_images(opposite_point, points, names, output_dir="B")

# Plot using saved clusters
plot_clusters_with_given_indices(
    base_point,
    opposite_point,
    points,
    base_indices,
    opp_indices,
    title=f"{target_angle}° - {target_angle+180}° Clusters",
    save_path=f"clusters_{target_angle}_{target_angle+180}.png",
)

# Generate rotation sequences
rotation_seq_A = rotate_and_find_nearest(
    base_point=base_point, all_points=points, all_names=names, num_steps=360
)
df_A = pd.DataFrame(rotation_seq_A, columns=["step", "angle_deg", "filename"])
df_A.to_csv("rotation_sequence_A.csv", index=False)
print("Saved rotation sequence A to rotation_sequence_A.csv")

rotation_seq_B = rotate_and_find_nearest(
    base_point=opposite_point, all_points=points, all_names=names, num_steps=360
)
df_B = pd.DataFrame(rotation_seq_B, columns=["step", "angle_deg", "filename"])
df_B.to_csv("rotation_sequence_B.csv", index=False)
print("Saved rotation sequence B to rotation_sequence_B.csv")

plot_two_rotation_paths_fixed_color(
    all_points=points,
    all_names=names,
    base_point=base_point,
    opposite_point=opposite_point,
    rotation_seq_A=rotation_seq_A,
    rotation_seq_B=rotation_seq_B,
    title="Rotation Paths Around Base and Opposite Points",
)


def save_rotation_comparison_series_with_plots(
    base_point,
    opposite_point,
    all_points,
    all_names,
    num_images=10,
    angle_step=45,
    image_source_dir="female_faces",
    output_root="rotation_pairs",
):
    os.makedirs(output_root, exist_ok=True)

    for angle in range(0, 180, angle_step):
        angle_str = f"{angle:03d}"
        dir_path = os.path.join(output_root, f"rotation_{angle_str}")
        os.makedirs(dir_path, exist_ok=True)
        dir_A = os.path.join(dir_path, "A")
        dir_B = os.path.join(dir_path, "B")
        os.makedirs(dir_A, exist_ok=True)
        os.makedirs(dir_B, exist_ok=True)

        # Rotate base and opposite points
        rot_A = rotate_vector(base_point, angle)
        rot_B = rotate_vector(opposite_point, angle)

        # Compute distances
        dists_A = np.linalg.norm(all_points - rot_A, axis=1)
        dists_B = np.linalg.norm(all_points - rot_B, axis=1)

        nearest_A = np.argsort(dists_A)[:num_images]
        nearest_B = np.argsort(dists_B)[:num_images]

        selected_A_names = []
        selected_B_names = []

        for idx in nearest_A:
            name = all_names[idx]
            selected_A_names.append(name)
            src = os.path.join(image_source_dir, name)
            dst = os.path.join(dir_A, name)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Missing image A: {src}")

        for idx in nearest_B:
            name = all_names[idx]
            selected_B_names.append(name)
            src = os.path.join(image_source_dir, name)
            dst = os.path.join(dir_B, name)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                print(f"Missing image B: {src}")

        # Save filenames to CSV
        pd.DataFrame(selected_A_names, columns=["filename"]).to_csv(
            os.path.join(dir_A, "A_filenames.csv"), index=False
        )
        pd.DataFrame(selected_B_names, columns=["filename"]).to_csv(
            os.path.join(dir_B, "B_filenames.csv"), index=False
        )

        # Plot the selected clusters
        plot_clusters_with_given_indices(
            base_point=rot_A,
            opposite_point=rot_B,
            all_points=all_points,
            base_indices=nearest_A,
            opp_indices=nearest_B,
            title=f"Rotation {angle}°: A vs B",
            save_path=os.path.join(dir_path, f"rotation_{angle_str}_plot.png"),
        )

        print(
            f"Saved rotation {angle_str}°: {len(nearest_A)} A + {len(nearest_B)} B images"
        )


save_rotation_comparison_series_with_plots(
    base_point=base_point,
    opposite_point=opposite_point,
    all_points=points,
    all_names=names,
    num_images=10,
    angle_step=45,
    image_source_dir="female_faces",
    output_root="rotation_pairs",
)
