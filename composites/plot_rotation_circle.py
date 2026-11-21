#################################################################
# plot_rotation_circle.py
#
# PCA circle plot in the style of create_prediction_scatter
# (classify_rotation_resnet50.py): all images in gray, dashed reference
# circle, spokes every 20 deg, PC1/PC2 axes - plus the rotation path used by
# make_rotation_composites.py and example composite faces placed around the
# circle at their angles.
#
# Run after make_rotation_composites.py, e.g.
#   py composites/plot_rotation_circle.py --every 30
#################################################################
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_crop(path, bg=128, pad=4):
    """Crop a masked frame to the face (non-background area) plus a margin."""
    img = np.asarray(Image.open(path).convert("RGB"))
    face = np.abs(img.astype(int) - bg).max(axis=2) > 3
    ys, xs = np.nonzero(face)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "pca_top2_filtered_female_vgg_1.csv"))
    ap.add_argument("--frames", default=str(ROOT / "composites" / "output"),
                    help="output folder of make_rotation_composites.py")
    ap.add_argument("--radius", type=float, default=0.45, help="rotation radius used for the frames")
    ap.add_argument("--every", type=float, default=30.0, help="show a face every N degrees")
    ap.add_argument("--zoom", type=float, default=0.62, help="size of the face thumbnails")
    ap.add_argument("--image-radius", type=float, default=1.22,
                    help="distance of the thumbnails from the centre, relative to the reference circle")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, header=None, names=["name", "x", "y"])
    frames_dir = Path(args.frames)
    meta = pd.read_csv(frames_dir / "frames.csv")

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.scatter(df["x"], df["y"], s=5, alpha=0.3, color="gray", label="All Vectors")

    # Reference circle and spokes, as in create_prediction_scatter.
    ref_radius = np.sqrt(df["x"] ** 2 + df["y"] ** 2).max() * 1.05
    ax.add_patch(Circle((0, 0), ref_radius, fill=False, color="black", linestyle="--", alpha=0.5))
    for a in range(0, 360, 20):
        r = np.deg2rad(a)
        x, y = ref_radius * np.cos(r), ref_radius * np.sin(r)
        ax.plot([0, x], [0, y], color="gray", linewidth=0.5, alpha=0.5)
        ax.text(x * 1.05, y * 1.05, f"{a}°", ha="center", va="center")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axvline(x=0, color="black", linewidth=1)

    # Rotation path of the composites.
    ax.add_patch(Circle((0, 0), args.radius, fill=False, color="black", linewidth=1.6,
                        label=f"Rotation path (r = {args.radius})"))

    # Example faces around the circle.
    img_r = ref_radius * args.image_radius
    for angle in np.arange(0, 360, args.every):
        step = int((meta["target_angle"] - angle).abs().idxmin())
        theta = meta.loc[step, "target_angle"]
        rad = np.deg2rad(theta)
        px, py = args.radius * np.cos(rad), args.radius * np.sin(rad)
        ix, iy = img_r * np.cos(rad), img_r * np.sin(rad)

        if np.isclose(theta, 0):
            color, label = "blue", "A (0°)"
        elif np.isclose(theta, 180):
            color, label = "red", "B (180°)"
        else:
            color, label = "black", None
        ax.scatter([px], [py], s=70, color=color, edgecolor="white", linewidth=1, zorder=5,
                   label=label)
        ax.plot([px, ix], [py, iy], color=color, linewidth=0.8, alpha=0.7, zorder=4)

        face = load_crop(frames_dir / "frames" / f"frame_{step:04d}.png")
        box = AnnotationBbox(OffsetImage(face, zoom=args.zoom), (ix, iy), frameon=True,
                             pad=0.15, zorder=6,
                             bboxprops=dict(edgecolor=color, linewidth=2 if label else 0.8))
        ax.add_artist(box)
        # Label on the outer side of the thumbnail, away from the spoke labels.
        label_r = img_r + 0.21 * ref_radius * args.zoom / 0.62
        ax.text(label_r * np.cos(rad), label_r * np.sin(rad), f"{theta:.0f}°",
                ha="center", va="center", fontsize=11, fontweight="bold", color=color, zorder=7)

    lim = img_r + 0.28 * ref_radius
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Composite faces along the rotation in PCA space")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = Path(args.out) if args.out else frames_dir / "rotation_circle.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
