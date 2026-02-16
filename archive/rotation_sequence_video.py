# make_rotation_video.py
# Creates a rotation-sequence CSV (closest image per angle step) and then builds a video from those images.
#
# Usage examples:
#   python make_rotation_video.py --pca_csv pca_top2_filtered_female.csv --image_dir female_faces --base_angle 0 --num_steps 180 --rotation_range 180 --fps 12
#   python make_rotation_video.py --base_angle 135 --rotation_range 360 --num_steps 360 --out_video rot_135.mp4
#
# Output:
#   - tmp/rotation_sequence.csv
#   - tmp/rotation_video.mp4  (or your --out_video)

import os
import argparse
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = "tmp"


def inside_tmp(*paths):
    os.makedirs(BASE_DIR, exist_ok=True)
    return os.path.join(BASE_DIR, *paths)


def load_top2_filtered(csv_path):
    """
    CSV format (no header): image_name, x, y
    """
    df = pd.read_csv(csv_path, header=None)
    names = df.iloc[:, 0].astype(str).values
    x = df.iloc[:, 1].astype(float).values
    y = df.iloc[:, 2].astype(float).values
    points = np.stack((x, y), axis=1)
    return names, points


def rotate_vector(v, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ],
        dtype=float,
    )
    return R @ v


def create_base_and_opposite_points(
    target_angle_deg, target_radius=0.45, pca_csv="pca_top2_filtered_female.csv"
):
    names, points = load_top2_filtered(pca_csv)
    angles = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) + 360) % 360
    radii = np.linalg.norm(points, axis=1)

    angle_err = np.abs(angles - target_angle_deg)
    angle_err = np.minimum(angle_err, 360 - angle_err)  # circular distance
    radius_err = np.abs(radii - target_radius)

    # weight radius strongly so we stay near the same ring (like your original)
    combined = angle_err + 100 * radius_err
    base_idx = int(np.argmin(combined))
    base_point = points[base_idx]
    opposite_point = -base_point
    return base_point, opposite_point


def generate_rotation_sequence(
    base_point,
    all_points,
    all_names,
    num_steps=180,
    start_angle=0.0,
    rotation_range=180.0,
    used_indices=None,
):
    """
    For each step, rotate base_point and pick nearest image in PCA-2D.
    Returns list of dict rows: step, angle_deg, filename
    """
    if used_indices is None:
        used_indices = set()

    results = []
    for i in range(num_steps + 1):
        angle_deg = (start_angle + (rotation_range * i / num_steps)) % 360
        rotated = rotate_vector(base_point, angle_deg)

        dists = np.linalg.norm(all_points - rotated, axis=1)
        if used_indices:
            for idx in used_indices:
                dists[idx] = np.inf

        idx_closest = int(np.argmin(dists))
        used_indices.add(idx_closest)

        true_angle = (np.degrees(np.arctan2(rotated[1], rotated[0])) + 360) % 360
        results.append(
            {
                "step": i,
                "angle_deg": float(true_angle),
                "filename": str(all_names[idx_closest]),
            }
        )
    return results, used_indices


def _load_font(size=20):
    # Best-effort: use default bitmap font if no TTF is available.
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def build_video(
    sequence_df,
    image_dir,
    out_video_path,
    fps=12,
    frame_size=512,
    annotate=True,
    missing="skip",  # "skip" or "black"
):
    """
    Writes an MP4 from the ordered filenames in sequence_df.
    Tries OpenCV first, falls back to imageio if OpenCV isn't installed.
    """
    # Try OpenCV writer
    try:
        import cv2  # type: ignore

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_video_path, fourcc, float(fps), (frame_size, frame_size)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter failed to open.")

        font = _load_font(18)

        for _, row in sequence_df.iterrows():
            fname = str(row["filename"])
            angle = float(row["angle_deg"])
            path = os.path.join(image_dir, fname)

            if not os.path.exists(path):
                if missing == "skip":
                    continue
                img = Image.new("RGB", (frame_size, frame_size), (0, 0, 0))
            else:
                img = (
                    Image.open(path)
                    .convert("RGB")
                    .resize((frame_size, frame_size), Image.BICUBIC)
                )

            if annotate:
                draw = ImageDraw.Draw(img)
                txt = f"{angle:.1f}° | {fname}"
                draw.rectangle([0, 0, frame_size, 28], fill=(0, 0, 0))
                draw.text((8, 5), txt, fill=(255, 255, 255), font=font)

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()
        return

    except Exception:
        pass

    # Fallback: imageio
    import imageio.v2 as imageio  # type: ignore

    font = _load_font(18)
    with imageio.get_writer(
        out_video_path, fps=fps, codec="libx264", quality=8
    ) as writer:
        for _, row in sequence_df.iterrows():
            fname = str(row["filename"])
            angle = float(row["angle_deg"])
            path = os.path.join(image_dir, fname)

            if not os.path.exists(path):
                if missing == "skip":
                    continue
                img = Image.new("RGB", (frame_size, frame_size), (0, 0, 0))
            else:
                img = (
                    Image.open(path)
                    .convert("RGB")
                    .resize((frame_size, frame_size), Image.BICUBIC)
                )

            if annotate:
                draw = ImageDraw.Draw(img)
                txt = f"{angle:.1f}° | {fname}"
                draw.rectangle([0, 0, frame_size, 28], fill=(0, 0, 0))
                draw.text((8, 5), txt, fill=(255, 255, 255), font=font)

            writer.append_data(np.array(img))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_csv", default="pca_top2_filtered_female.csv")
    ap.add_argument("--image_dir", default="female_faces")

    ap.add_argument(
        "--base_angle",
        type=float,
        default=0.0,
        help="Angle (deg) to pick the initial base point from the PCA set.",
    )
    ap.add_argument(
        "--target_radius",
        type=float,
        default=0.45,
        help="Ring radius preference when selecting base point.",
    )

    ap.add_argument("--num_steps", type=int, default=360)
    ap.add_argument("--start_angle", type=float, default=0.0)
    ap.add_argument("--rotation_range", type=float, default=360.0)

    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frame_size", type=int, default=512)
    ap.add_argument(
        "--annotate", action="store_true", help="Overlay angle + filename on frames."
    )
    ap.add_argument("--missing", choices=["skip", "black"], default="skip")

    ap.add_argument("--out_csv", default=inside_tmp("rotation_sequence.csv"))
    ap.add_argument("--out_video", default=inside_tmp("rotation_video.mp4"))

    ap.add_argument(
        "--clean_tmp", action="store_true", help="Delete tmp/ before writing outputs."
    )
    args = ap.parse_args()

    if args.clean_tmp and os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR, ignore_errors=True)
        os.makedirs(BASE_DIR, exist_ok=True)

    names, points = load_top2_filtered(args.pca_csv)
    base_point, _ = create_base_and_opposite_points(
        target_angle_deg=args.base_angle,
        target_radius=args.target_radius,
        pca_csv=args.pca_csv,
    )

    seq_rows, _ = generate_rotation_sequence(
        base_point=base_point,
        all_points=points,
        all_names=names,
        num_steps=args.num_steps,
        start_angle=args.start_angle,
        rotation_range=args.rotation_range,
        used_indices=set(),  # avoid repeats within the sequence
    )

    df_seq = pd.DataFrame(seq_rows, columns=["step", "angle_deg", "filename"])
    df_seq.to_csv(args.out_csv, index=False)
    print(f"Saved rotation sequence CSV: {args.out_csv} (rows={len(df_seq)})")

    build_video(
        sequence_df=df_seq,
        image_dir=args.image_dir,
        out_video_path=args.out_video,
        fps=args.fps,
        frame_size=args.frame_size,
        annotate=args.annotate,
        missing=args.missing,
    )
    build_image_sequence_folder(
        sequence_df=df_seq,
        image_dir=args.image_dir,
        out_dir=inside_tmp("rotation_frames"),
        frame_size=args.frame_size,
        annotate=args.annotate,
        missing=args.missing,
    )

    print(f"Saved video: {args.out_video}")


def build_image_sequence_folder(
    sequence_df,
    image_dir,
    out_dir="rotation_frames",
    frame_size=512,
    annotate=True,
    missing="skip",  # "skip" or "black"
):
    """
    Creates a folder with ordered images according to rotation sequence.
    Files will be named 0000.png, 0001.png, ...
    """

    os.makedirs(out_dir, exist_ok=True)
    font = _load_font(18)

    for i, row in sequence_df.iterrows():
        fname = str(row["filename"])
        angle = float(row["angle_deg"])
        path = os.path.join(image_dir, fname)

        if not os.path.exists(path):
            if missing == "skip":
                continue
            img = Image.new("RGB", (frame_size, frame_size), (0, 0, 0))
        else:
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((frame_size, frame_size), Image.BICUBIC)
            )

        if annotate:
            draw = ImageDraw.Draw(img)
            txt = f"{angle:.1f}° | {fname}"
            draw.rectangle([0, 0, frame_size, 28], fill=(0, 0, 0))
            draw.text((8, 5), txt, fill=(255, 255, 255), font=font)

        out_path = os.path.join(out_dir, f"{i:04d}.png")
        img.save(out_path)

    print(f"Saved {len(sequence_df)} frames to folder: {out_dir}")


if __name__ == "__main__":
    main()
