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


import numpy as np
import pandas as pd


def load_top2_filtered(csv_path="pca_top2_filtered.csv"):
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


def rotate_vector(v, angle_deg):
    """
    Rotate a 2D vector by angle_deg (in degrees) around the origin.

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
