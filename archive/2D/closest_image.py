#################################################################
# This script loads a 2D PCA representation of images, selects a random point,
# computes the closest point to it, and copies both images to a new folder.
# the goal is to see if the images are visually similar.
#################################################################


import pandas as pd
import numpy as np
import os
import shutil

# Load data
df = pd.read_csv("pca_2D_24.csv", header=None)
names = df.iloc[:, 0].to_numpy()
x = df.iloc[:, 1].to_numpy()
y = df.iloc[:, 2].to_numpy()
points = np.stack((x, y), axis=1)

# Choose one random index
random_idx = np.random.choice(len(points))
random_point = points[random_idx]

# Compute distances to all other points
dists = np.linalg.norm(points - random_point, axis=1)
dists[random_idx] = np.inf  # exclude self

# Get index of closest point
closest_idx = np.argmin(dists)

# Get names
selected_names = [names[random_idx], names[closest_idx]]

# Print what was selected
print("Random image selected:", selected_names[0])
print("Closest image to it:", selected_names[1])

# Copy images to new folder
source_folder = "celebA"
dest_folder = "random_and_closest"
os.makedirs(dest_folder, exist_ok=True)

for name in selected_names:
    src = os.path.join(source_folder, name)
    dst = os.path.join(dest_folder, name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"Warning: file not found: {src}")
