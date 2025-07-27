#################################################################
# This script starts from a random 2D vector (image embedding)
# and walks step-by-step to the closest unused point in the dataset.
# For each step, it saves a side-by-side frame: scatter with current point,
# and the image corresponding to the closest point.
# Finally, it compiles these frames into a video.
#### not very efficient
#################################################################

import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.image import imread

# Load the 2D embeddings (after PCA)
df = pd.read_csv("pca_top2_filtered.csv", header=None)
names = df.iloc[:, 0].to_numpy()  # image filenames
x = df.iloc[:, 1].to_numpy()  # PCA dimension 1
y = df.iloc[:, 2].to_numpy()  # PCA dimension 2
points = np.stack((x, y), axis=1)  # shape: (N, 2)


# Choose a random starting point
def choose_base_point():
    idx = np.random.choice(len(points))
    base_point = points[idx]
    base_name = names[idx]
    return base_point, base_name, idx


# Draw the selected base point on top of the full scatter
def draw_base_point(base_point):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(x, y, color="lightgray", s=10, label="All points")
    ax.scatter(*base_point, color="blue", s=10, label="Base point")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Selected Base Point for Walk")
    ax.legend()
    plt.show()


# Find the closest point to current_vec, ignoring excluded indices
def closest_point(current_vec, points, names, exclude_idx=None):
    dists = np.linalg.norm(points - current_vec, axis=1)
    if exclude_idx is not None:
        dists[exclude_idx] = np.inf  # mask already used indices
    closest_idx = int(np.argmin(dists))
    return closest_idx, names[closest_idx], dists[closest_idx]


# Create side-by-side frame of scatter + image
def save_dual_frame(step_num, closest_idx, closest_name, frame_path):
    img_path = os.path.join("celebA", closest_name)
    if not os.path.exists(img_path):
        print(f"Warning: {closest_name} not found")
        return False
    img = imread(img_path)

    fig, (ax_scatter, ax_img) = plt.subplots(1, 2, figsize=(6, 6))

    # Scatter plot with current point
    ax_scatter.scatter(x, y, color="lightgray", s=10, zorder=1)
    ax_scatter.scatter(
        points[closest_idx, 0],
        points[closest_idx, 1],
        color="blue",
        s=10,
        zorder=2,
        label="Current point",
    )
    ax_scatter.set_title(f"Step {step_num}", fontsize=9)
    ax_scatter.axhline(0, color="black", linewidth=0.5)
    ax_scatter.axvline(0, color="black", linewidth=0.5)
    ax_scatter.set_aspect("equal")
    ax_scatter.grid(True)
    ax_scatter.legend()

    # Show image of the closest point
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(closest_name, fontsize=8)

    plt.tight_layout()
    fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


# ---------------- MAIN WALK ---------------- #

ring = "walk"  # label used for filenames
base_point, base_name, base_idx = choose_base_point()
draw_base_point(base_point)

# Prepare folder to hold the video frames
temp_frame_folder = f"temp_walk_frames_{ring}"
os.makedirs(temp_frame_folder, exist_ok=True)

closest_names = []
used_indices = set([base_idx])  # avoid revisiting the base point
current_vec = base_point

num_steps = 1000  # number of walk steps (or limit to len(points))

plt.ioff()  # turn off matplotlib interactive mode

for step in range(num_steps):
    exclude_idx = list(used_indices)
    closest_idx, closest_name, dist = closest_point(
        current_vec, points, names, exclude_idx=exclude_idx
    )

    used_indices.add(closest_idx)
    closest_names.append(closest_name)

    frame_path = os.path.join(temp_frame_folder, f"frame_{step:04d}.jpg")
    save_dual_frame(step, closest_idx, closest_name, frame_path)

    # Update current location to the new point
    current_vec = points[closest_idx]

# ---------------- VIDEO COMPILATION ---------------- #

video_path = f"walk_closest_points_{ring}.mp4"
first_frame = cv2.imread(os.path.join(temp_frame_folder, "frame_0000.jpg"))
height, width, _ = first_frame.shape
video_writer = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)
)

for step in range(num_steps):
    frame_path = os.path.join(temp_frame_folder, f"frame_{step:04d}.jpg")
    img = cv2.imread(frame_path)
    if img is not None:
        video_writer.write(img)

video_writer.release()
print(f"Video saved as {video_path}")
