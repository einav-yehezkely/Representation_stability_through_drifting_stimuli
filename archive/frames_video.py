import os
import cv2

# ============================================================
# SETTINGS
# ============================================================

FRAMES_DIR = "C:\Users\einav\clone_here\Representation_stability_through_drifting_stimuli\output_CE\linear_frames"
OUTPUT_VIDEO = "linear_frames_video.mp4"

FPS = 30

# ============================================================
# GET FRAMES
# ============================================================

frame_files = sorted([
    f for f in os.listdir(FRAMES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

if not frame_files:
    raise ValueError("No image frames found in the folder.")

print(f"Found {len(frame_files)} frames")

# ============================================================
# GET FRAME SIZE
# ============================================================

first_frame_path = os.path.join(FRAMES_DIR, frame_files[0])
first_frame = cv2.imread(first_frame_path)

if first_frame is None:
    raise ValueError(f"Could not read: {first_frame_path}")

height, width, _ = first_frame.shape

# ============================================================
# CREATE VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height)
)

for i, filename in enumerate(frame_files):

    frame_path = os.path.join(FRAMES_DIR, filename)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Skipping {filename}")
        continue

    # Resize in case some frames have a slightly different size
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height))

    video.write(frame)

    if i % 100 == 0:
        print(f"{i}/{len(frame_files)}")

video.release()

print(f"Video saved to: {OUTPUT_VIDEO}")