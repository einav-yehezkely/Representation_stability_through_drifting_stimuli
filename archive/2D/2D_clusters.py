#################################################################
# This script loads a 2D PCA representation of images, selects random points in the first and third quadrants,
# computes the 20 closest points to these random points, saves the results in CSV files
# and copies the corresponding images to new folders.
# It also plots the points and the clusters.
# The goal is to see if the clusters are visually distinct and if the images in the clusters are similar.
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Load vectors
df = pd.read_csv("pca_top2_filtered_female.csv", header=None)
names = df.iloc[:, 0].to_numpy()
x = df.iloc[:, 1].to_numpy()
y = df.iloc[:, 2].to_numpy()
points = np.stack((x, y), axis=1)

# Define masks for Q1 and Q3
mask_q1 = (x > 0) & (y > 0)
mask_q3 = (x < 0) & (y < 0)

# Choose random point from Q1 and Q3
points_q1 = points[mask_q1]
points_q3 = points[mask_q3]
names_q1 = names[mask_q1]
names_q3 = names[mask_q3]

idx_q1 = np.random.choice(len(points_q1))
idx_q3 = np.random.choice(len(points_q3))
center_q1 = points_q1[idx_q1]
center_q3 = points_q3[idx_q3]

# Compute distances from all points to each center
dists_to_q1 = np.linalg.norm(points - center_q1, axis=1)
dists_to_q3 = np.linalg.norm(points - center_q3, axis=1)

# Take 20 closest
indices_q1 = np.argsort(dists_to_q1)[:20]
indices_q3 = np.argsort(dists_to_q3)[:20]

# Extract data
cluster_q1 = df.iloc[indices_q1]
cluster_q3 = df.iloc[indices_q3]

# Save CSVs
# cluster_q1.to_csv("random_cluster_q1.csv", index=False, header=False)
# cluster_q3.to_csv("random_cluster_q3.csv", index=False, header=False)

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color="lightgray", s=10)
plt.scatter(
    x[indices_q1], y[indices_q1], color="red", label="Closest to random Q1 point"
)
plt.scatter(
    x[indices_q3], y[indices_q3], color="blue", label="Closest to random Q3 point"
)
plt.scatter(*center_q1, color="black", marker="x", label="Random Q1 center")
plt.scatter(*center_q3, color="black", marker="x", label="Random Q3 center")
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.title("Random Q1 & Q3 Clusters (20 closest)")
plt.show()

# Copy images
source_folder = "celebA"

for name, indices in [("q1_random", indices_q1), ("q3_random", indices_q3)]:
    dest_folder = f"{name}_images"
    os.makedirs(dest_folder, exist_ok=True)
    for idx in indices:
        img_name = names[idx]
        src = os.path.join(source_folder, img_name)
        dst = os.path.join(dest_folder, img_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {img_name} not found.")
