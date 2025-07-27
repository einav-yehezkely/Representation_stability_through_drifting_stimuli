import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from generate_rotation_sequence import *

# load points
names, points = load_top2_filtered("files/pca_top2_filtered_10per.csv")

# choose a base point A (image)
A_base_index = np.random.choice(len(points))  # Randomly select a base index
A_base_point = points[A_base_index]
A_base_name = names[A_base_index]


def find_opposite(base, all_points):
    opposite_vec = -base
    dists = np.linalg.norm(all_points - opposite_vec, axis=1)
    return np.argmin(dists)


# choose the opposite point B (image)
B_base_index = find_opposite(A_base_point, points)
B_base_point = points[B_base_index]
B_base_name = names[B_base_index]

# create trajectories from A and B
trajectory_A = rotate_and_find_nearest(A_base_point, points, names, num_steps=1000)
trajectory_B = rotate_and_find_nearest(B_base_point, points, names, num_steps=1000)

image_names_A = [name for (_, _, name) in trajectory_A]
image_names_B = [name for (_, _, name) in trajectory_B]


def create_random_order(image_names_A, image_names_B):
    """
    Create a random order of images from two groups A and B.
    """

    all_images = image_names_A + image_names_B
    np.random.shuffle(all_images)
    images_to_present = all_images
    return images_to_present


images_to_present = create_random_order(image_names_A, image_names_B)

groups = {"A": [], "B": []}

for i in range(len(images_to_present)):
    if i == 1:
        groups["A"].append(images_to_present[i])
