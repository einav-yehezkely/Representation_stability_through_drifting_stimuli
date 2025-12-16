#################################################################
# This script rotates a vector in 2D space and finds the closest point (aka image)
# from a set of points for each angle of rotation. It then creates a video
# showing the rotation and the closest point at each angle.
#################################################################

import pandas as pd
import numpy as np
import os
import cv2  # OpenCV
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.animation import FuncAnimation


# Load vectors
df = pd.read_csv("pca_top2_filtered.csv", header=None)
names = df.iloc[:, 0].to_numpy()
x = df.iloc[:, 1].to_numpy()
y = df.iloc[:, 2].to_numpy()
points = np.stack((x, y), axis=1)


def choose_points_from_rings():
    # compute distances from origin
    r = np.linalg.norm(points, axis=1)
    r_max = np.max(r)

    # Define masks for Q1
    mask_q1 = (x > 0) & (y > 0)

    # Define inner and outer rings for Q1
    inner_min = 0.10 * r_max
    inner_max = 0.15 * r_max
    mask_inner_ring_q1 = mask_q1 & (r >= inner_min) & (r <= inner_max)

    outer_min = 0.30 * r_max
    outer_max = 0.35 * r_max
    mask_outer_ring_q1 = mask_q1 & (r >= outer_min) & (r <= outer_max)

    # Choose random points from inner and outer rings of Q1
    idx_inner = np.random.choice(np.where(mask_inner_ring_q1)[0])
    idx_outer = np.random.choice(np.where(mask_outer_ring_q1)[0])

    point_inner = points[idx_inner]
    point_outer = points[idx_outer]

    name_inner = names[idx_inner]
    name_outer = names[idx_outer]

    return (
        point_inner,
        point_outer,
        inner_min,
        inner_max,
        outer_min,
        outer_max,
        name_inner,
        name_outer,
    )


# draw the points and rings
def draw_circle(ax, radius, **kwargs):
    circle = plt.Circle((0, 0), radius, fill=False, **kwargs)
    ax.add_patch(circle)


def draw_rings_and_points():
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.scatter(x, y, color="lightgray", s=10)
    ax.scatter(*point_inner, color="green", s=10)
    ax.scatter(*point_outer, color="red", s=10)

    draw_circle(ax, inner_min, color="green", linestyle="--", linewidth=1)
    draw_circle(ax, inner_max, color="green", linestyle="--", linewidth=1)
    draw_circle(ax, outer_min, color="red", linestyle="--", linewidth=1)
    draw_circle(ax, outer_max, color="red", linestyle="--", linewidth=1)

    ax.axhline(0, color="black")
    ax.axvline(0, color="black")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title("Random Points from Two Concentric Rings in Q1")
    plt.show()


(
    point_inner,
    point_outer,
    inner_min,
    inner_max,
    outer_min,
    outer_max,
    name_inner,
    name_outer,
) = choose_points_from_rings()
draw_rings_and_points()
plt.show()


def closest_point(rotated_vec, points, names, exclude_idx=None):
    dists = np.linalg.norm(points - rotated_vec, axis=1)
    if exclude_idx is not None:
        dists[exclude_idx] = np.inf
    closest_idx = int(np.argmin(dists))
    return closest_idx, names[closest_idx], dists[closest_idx]


def rotate_vector(vec, angle_deg, clockwise=True):
    if clockwise:
        angle_deg = -angle_deg
    theta = np.radians(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ vec


def save_dual_frame(
    angle_deg,
    closest_idx,
    closest_name,
    inner_min,
    inner_max,
    outer_min,
    outer_max,
    frame_path,
    ring,
):

    img_path = os.path.join("celebA", closest_name)
    if not os.path.exists(img_path):
        print(f"Warning: {closest_name} not found")
        return False
    img = imread(img_path)

    fig, (ax_scatter, ax_img) = plt.subplots(1, 2, figsize=(6, 6))

    # left plot: scatter plot with rings
    ax_scatter.scatter(x, y, color="lightgray", s=10, zorder=1)

    color_ = "green" if ring == "inner" else "red"

    if ring == "inner":
        draw_circle(ax_scatter, inner_min, color="green", linestyle="--", linewidth=1)
        draw_circle(ax_scatter, inner_max, color="green", linestyle="--", linewidth=1)
    else:
        draw_circle(ax_scatter, outer_min, color="red", linestyle="--", linewidth=1)
        draw_circle(ax_scatter, outer_max, color="red", linestyle="--", linewidth=1)

    ax_scatter.scatter(
        points[closest_idx, 0], points[closest_idx, 1], color=color_, s=10, zorder=2
    )
    ax_scatter.set_title(f"Angle {angle_deg}°", fontsize=9)
    ax_scatter.axhline(0, color="black")
    ax_scatter.axvline(0, color="black")
    ax_scatter.grid(True)
    ax_scatter.set_aspect("equal")

    # right plot: image of the closest point
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(closest_name, fontsize=8)

    plt.tight_layout()
    fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


ring = "outer"  # Specify which ring to use: "inner" or "outer"
base_vector = point_inner if ring == "inner" else point_outer

# Create a temporary folder for frames
temp_frame_folder = f"temp_rotation_frames_{ring}"
os.makedirs(temp_frame_folder, exist_ok=True)

# Initialize a list to store the closest name for each angle
closest_names = []

# Prepare for animation
plt.ioff()

prev_idx = None

angle_step = 1
num_frames = int(360 / angle_step)

# Rotate the base vector and find the closest image for each angle
for i in range(num_frames):
    angle = i * angle_step
    rotated_vec = rotate_vector(base_vector, angle)
    closest_idx, closest_name, dist = closest_point(
        rotated_vec, points, names, exclude_idx=prev_idx
    )

    closest_names.append(closest_name)

    frame_path = os.path.join(temp_frame_folder, f"frame_{angle:04d}.jpg")
    save_dual_frame(
        angle,
        closest_idx,
        closest_name,
        inner_min,
        inner_max,
        outer_min,
        outer_max,
        frame_path,
        ring,
    )

    prev_idx = closest_idx

# Create a video from the saved frames
video_path = f"rotating_vector_{ring}_ring.mp4"
first_frame = cv2.imread(os.path.join(temp_frame_folder, "frame_0000.jpg"))
height, width, _ = first_frame.shape
video_writer = cv2.VideoWriter(
    video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)
)

for angle in range(360):
    frame_path = os.path.join(temp_frame_folder, f"frame_{angle:03d}.jpg")
    img = cv2.imread(frame_path)
    if img is not None:
        video_writer.write(img)

video_writer.release()
print(f"Video saved as{video_path}")
