"""
Verbatim copies of the scientific functions used by the neural-network experiment.

Copied (unchanged bodies) from human_experiment/reference_source/classify_rotation_resnet50.py,
which is itself a byte-identical copy of network_classification/classify_rotation_resnet50.py.
The original modules execute training code at import time (torch, CSV loading), so they cannot be
imported directly; the functions below are copied so they can be called in isolation.

Source line numbers refer to network_classification/classify_rotation_resnet50.py at commit 48cadce.
"""

import numpy as np
import pandas as pd


# classify_rotation_resnet50.py:236-251
def load_top2_filtered(csv_path):
    """
    Load 2D PCA coordinates of filtered images from a CSV file.

    Assumes the format: image_name, x, y
    """
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    points = np.stack((x, y), axis=1)
    return names, points


# classify_rotation_resnet50.py:255-297
def create_base_and_opposite_points(angle, csv_path):
    """
    Given a target angle, find the base point in PCA space that is closest to that angle,
    and compute its opposite point (180 degrees away).
    """
    # Load data
    names, points = load_top2_filtered(csv_path)

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

    # Compute a combined error metric that considers both angle and radius differences
    delta = np.abs(
        angles_deg - target_angle
    )

    angle_error = np.minimum(
        delta,
        360 - delta
    )
    radius_error = np.abs(radii - target_radius)
    combined_error = angle_error + radius_error * 100

    # Find the index of the point whose angle is closest to the target angle
    base_idx = np.argmin(combined_error)
    # Retrieve the actual 2D PCA coordinates of the selected base point
    base_point = points[base_idx]

    opposite_point = -base_point

    return base_point, opposite_point


# classify_rotation_resnet50.py:300-318
def rotate_vector(v, angle_deg):
    """
    Rotate a 2D vector counter clockwise by angle_deg (in degrees) around the origin.
    """
    angle_rad = np.deg2rad(angle_deg)
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return R @ v


# classify_rotation_resnet50.py:321-357 (file output arguments kept, unused, as in the source)
def collect_nearest_images(
    center_point,
    all_points,
    all_names,
    output_dir=None,
    k=500,
    image_source_dir="female_faces",
):
    """
    Find the k nearest images to center_point and save their filenames.
    """

    # Compute distances
    dists = np.linalg.norm(all_points - center_point, axis=1)

    # Use np.argpartition for efficiency, then sort the selected indices
    nearest_indices = np.argpartition(dists, k)[:k]
    nearest_indices = nearest_indices[np.argsort(dists[nearest_indices])]

    selected_names = []

    for idx in nearest_indices:
        name = all_names[idx]
        selected_names.append(name)

    return nearest_indices, selected_names


# classify_rotation_resnet50.py:760-804
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
    """
    results = []
    if used_indices is None:
        used_indices = set()

    for i in range(num_steps):
        # Calculate the current angle for this step
        angle_deg = (start_angle + (rotation_range * i / num_steps)) % 360
        rotated = rotate_vector(base_point, angle_deg)
        true_angle = np.degrees(np.arctan2(rotated[1], rotated[0])) % 360
        dists = np.linalg.norm(all_points - rotated, axis=1)

        # Mark already used indices with infinite distance
        for idx in used_indices:
            dists[idx] = np.inf  # Ignore already used indices

        # Find the index of the closest point to the rotated position
        idx_closest = np.argmin(dists)
        used_indices.add(idx_closest)
        results.append((i, true_angle, all_names[idx_closest]))
    return results, used_indices
