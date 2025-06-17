import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_rotation_sequence import load_top2_filtered, rotate_and_find_nearest
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm  # progress bar


def create_video_from_sequence(image_names, image_folder, output_path, fps=10):
    """
    Create a video from a list of image filenames (in order).

    Parameters:
    - image_names: list of filenames (e.g., ["000123.jpg", ...])
    - image_folder: folder where the images are stored (e.g., "celebA")
    - output_path: path to the output video file (e.g., "rotation_video.mp4")
    - fps: frames per second in the output video
    """
    frames = []
    print("Loading images...")

    for name in tqdm(image_names):
        path = os.path.join(image_folder, name)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        img = Image.open(path).convert("RGB")
        img = img.resize((160, 160))  # resize for consistency
        frames.append(np.array(img))

    if not frames:
        print("No frames loaded. Aborting video creation.")
        return

    # Get frame size from first frame
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print("Creating video...")
    for frame in tqdm(frames):
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
    print(f"Video saved to: {output_path}")


# Step 1: Load the 2D PCA points
names, points = load_top2_filtered("files/pca_top2_filtered.csv")

# Step 2: Choose a base image (by index)
base_index = 123
base_point = points[base_index]
base_name = names[base_index]

# Step 3: Generate the rotation trajectory (1000 steps = 0.36° per step)
trajectory = rotate_and_find_nearest(base_point, points, names, num_steps=10000)

# Step 4: Extract the image names along the trajectory
image_names = [name for (_, _, name) in trajectory]

# Step 5: Create a video showing the rotation
create_video_from_sequence(
    image_names=image_names,
    image_folder="celebA",  # folder with the original images
    output_path="rotation_video.mp4",
    fps=5,  # adjust for speed of playback
)
