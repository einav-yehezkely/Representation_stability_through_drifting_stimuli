#################################################################
# make_rotation_video.py
#
# MP4 of the composite-face rotation: the face, large, on the left; on the
# right the PCA circle (all images in gray + rotation path) with a dot that
# moves to the current angle.
#
# Run after make_rotation_composites.py, e.g.
#   py composites/make_rotation_video.py --fps 30 --loops 2
#################################################################
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BG = (128, 128, 128)


def face_box(frames_dir, bg=128, pad=6):
    """Bounding box of the face (non-background pixels) over a few frames."""
    boxes = []
    for f in sorted((frames_dir / "frames").glob("frame_*.png"))[::30]:
        img = np.asarray(Image.open(f).convert("RGB")).astype(int)
        ys, xs = np.nonzero(np.abs(img - bg).max(axis=2) > 3)
        boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
    b = np.array(boxes)
    h, w = img.shape[:2]
    return (max(b[:, 0].min() - pad, 0), min(b[:, 1].max() + pad + 1, h),
            max(b[:, 2].min() - pad, 0), min(b[:, 3].max() + pad + 1, w))


def circle_panel(csv, radius, size):
    """Static PCA panel; returns (RGB image, function mapping (x, y) -> pixel)."""
    df = pd.read_csv(csv, header=None, names=["name", "x", "y"])
    ref = np.sqrt(df["x"] ** 2 + df["y"] ** 2).max() * 1.05
    lim = ref * 1.18

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(df["x"], df["y"], s=2, alpha=0.3, color="gray")
    ax.add_patch(Circle((0, 0), ref, fill=False, color="black", linestyle="--", alpha=0.5))
    for a in range(0, 360, 30):
        r = np.deg2rad(a)
        ax.plot([0, ref * np.cos(r)], [0, ref * np.sin(r)], color="gray", lw=0.5, alpha=0.5)
        ax.text(ref * 1.09 * np.cos(r), ref * 1.09 * np.sin(r), f"{a}°",
                ha="center", va="center", fontsize=11)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.add_patch(Circle((0, 0), radius, fill=False, color="black", lw=2))
    ax.plot([radius], [0], "o", color="blue", ms=9)
    ax.plot([-radius], [0], "o", color="red", ms=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    def to_px(x, y):
        return (int(round((x + lim) / (2 * lim) * size)),
                int(round((lim - y) / (2 * lim) * size)))

    return panel, to_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "composites" / "output"),
                    help="output folder of make_rotation_composites.py")
    ap.add_argument("--csv", default=str(ROOT / "pca_top2_filtered_female_vgg_1.csv"))
    ap.add_argument("--radius", type=float, default=0.45)
    ap.add_argument("--height", type=int, default=720, help="video height in px")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--loops", type=int, default=1, help="how many times to go around")
    ap.add_argument("--face-only", action="store_true", help="only the face, no PCA panel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    meta = pd.read_csv(frames_dir / "frames.csv")
    y0, y1, x0, x1 = face_box(frames_dir)

    H = args.height - args.height % 16
    face_h = H - 80
    face_w = int(round(face_h * (x1 - x0) / (y1 - y0)))
    face_w += face_w % 2
    left_w = face_w + 80
    panel_size = H - 40
    W = left_w if args.face_only else left_w + panel_size + 40
    W += (-W) % 16

    if not args.face_only:
        panel, to_px = circle_panel(args.csv, args.radius, panel_size)

    out = Path(args.out) if args.out else frames_dir / "rotation.mp4"
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=9,
                                pixelformat="yuv420p", macro_block_size=16)
    for _ in range(args.loops):
        for step, row in meta.iterrows():
            img = np.asarray(Image.open(frames_dir / "frames" / f"frame_{step:04d}.png").convert("RGB"))
            face = cv2.resize(img[y0:y1, x0:x1], (face_w, face_h), interpolation=cv2.INTER_CUBIC)

            canvas = np.full((H, W, 3), BG, np.uint8)
            top = (H - face_h) // 2
            canvas[top:top + face_h, 40:40 + face_w] = face

            theta = row["target_angle"]
            cv2.putText(canvas, f"{theta:5.1f} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 2, cv2.LINE_AA)

            if not args.face_only:
                p = panel.copy()
                rad = np.deg2rad(theta)
                cx, cy = to_px(0, 0)
                px, py = to_px(args.radius * np.cos(rad), args.radius * np.sin(rad))
                cv2.line(p, (cx, cy), (px, py), (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(p, (px, py), 13, (255, 165, 0), -1, cv2.LINE_AA)
                cv2.circle(p, (px, py), 13, (0, 0, 0), 2, cv2.LINE_AA)
                py0 = (H - panel_size) // 2
                canvas[py0:py0 + panel_size, left_w + 20:left_w + 20 + panel_size] = p
            writer.append_data(canvas)
    writer.close()
    print(f"Saved {out} ({W}x{H}, {len(meta) * args.loops / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
