import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from generate_rotation_sequence import *
import os
import shutil

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

# Define source and destination folders
source_folder = "celebA"  # Folder where the images are stored
destination_folder = "selected_images"
os.makedirs(destination_folder, exist_ok=True)  # Create folder if it doesn't exist

# Get source paths
A_image_path = os.path.join(source_folder, A_base_name)
B_image_path = os.path.join(source_folder, B_base_name)

# Get destination paths
A_dest_path = os.path.join(destination_folder, f"A_{A_base_name}")
B_dest_path = os.path.join(destination_folder, f"B_{B_base_name}")

# Copy files
shutil.copy(A_image_path, A_dest_path)
shutil.copy(B_image_path, B_dest_path)

print(f"Saved A image to {A_dest_path}")
print(f"Saved B image to {B_dest_path}")
