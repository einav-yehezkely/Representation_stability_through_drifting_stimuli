#################################################################
# This script loads a 2D PCA representation of images, selects slices of the data in each quadrant
# based on angles, and computes the points in those slices.
# It then saves the filtered points to CSV files and copies the corresponding images to new folders.
# The script also plots the points in 2D space, showing the slices in different colors.
# The goal is to see if the slices are visually distinct and if the images in the slices are similar.
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

df = pd.read_csv("pca_top2_filtered_female.csv", header=None)

# first column = names, second column = dimension 1, third column = dimension 2
x = df.iloc[:, 1]
y = df.iloc[:, 2]

mask_quadrant1 = (x > 0) & (y > 0)

# Calculate angles in degrees
angles = np.degrees(np.arctan2(y, x))
angles = (angles + 360) % 360

slices = {
    "Q1": {"mask": (x > 0) & (y > 0), "angle_min": 45, "angle_max": 60},
    "Q2": {"mask": (x < 0) & (y > 0), "angle_min": 135, "angle_max": 150},
    "Q3": {"mask": (x < 0) & (y < 0), "angle_min": 225, "angle_max": 240},
    "Q4": {"mask": (x > 0) & (y < 0), "angle_min": 315, "angle_max": 330},
}

source_folder = "celebA"

plt.figure(figsize=(7, 7), num="2D Image Vectors with Slices 24396 images")
# plt.figure(figsize=(7, 7), num="2D Image Vectors with Slices 997 images")

plt.scatter(x, y, color="lightgray", s=10, label="All Vectors")

for quadrant, info in slices.items():
    angle_min = info["angle_min"]
    angle_max = info["angle_max"]
    mask = info["mask"] & (angles >= angle_min) & (angles <= angle_max)
    filtered = df[mask]

    # Plot slice
    plt.scatter(
        x[mask], y[mask], s=10, label=f"{quadrant} Slice: {angle_min}°–{angle_max}°"
    )

    # # Save CSV
    # csv_name = f"{quadrant}_slice_{angle_min}_{angle_max}.csv"
    # filtered.to_csv(csv_name, index=False, header=False)
    # print(f"Saved slice to {csv_name}")

    # # Copy images
    # image_names = filtered.iloc[:, 0].tolist()
    # destination_folder = f"{angle_min}to{angle_max}degrees"
    # os.makedirs(destination_folder, exist_ok=True)

    # for name in image_names:
    #     src_path = os.path.join(source_folder, name)
    #     dst_path = os.path.join(destination_folder, name)
    #     if os.path.exists(src_path):
    #         shutil.copy2(src_path, dst_path)
    #     else:
    #         print(f"Warning: file not found: {src_path}")

# Final plot
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Image Vectors with Slices in All Quadrants - 24,396 images")
# plt.title("2D Image Vectors with Slices in All Quadrants - 997 images")
plt.axis("equal")
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
