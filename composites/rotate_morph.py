#################################################################
# rotate_morph.py
#
# Full-circle "rotation" morph through N anchor face images placed at
# known angles (e.g. 0, 90, 180, 270), using only those images.
#
# Between two neighbouring anchors k and k+1 (angular gap D, current
# offset phi) the frame is a spherical interpolation around the mean
# face M of all anchors:
#
#     w1 = sin(D - phi) / sin(D)      w2 = sin(phi) / sin(D)
#     X(theta) = M + w1 * (X_k - M) + w2 * (X_k+1 - M)
#
# For D = 90 this is w1 = cos(phi), w2 = sin(phi) - exactly how a 2D
# vector rotates (cos * v0 + sin * v90), so the distance from the mean
# stays constant instead of cutting through the centre like a straight
# morph. --mode linear gives the plain piecewise A->B morph instead.
#
# X is applied both to shape (landmarks) and to texture (pixels after
# warping every image to the current shape).
#################################################################
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import Delaunay

from morph_two import DEFAULT_MODEL, border_points, detect_landmarks, oval_mask, warp_to

HERE = Path(__file__).resolve().parent


def weights(phi, gap, mode):
    """Weights of (anchor k, anchor k+1, mean)."""
    if mode == "linear":
        t = phi / gap
        return 1 - t, t, 0.0
    g, p = np.deg2rad(gap), np.deg2rad(phi)
    w1, w2 = np.sin(g - p) / np.sin(g), np.sin(p) / np.sin(g)
    return w1, w2, 1 - w1 - w2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="anchor images, in angle order")
    ap.add_argument("--angles", type=float, nargs="+",
                    help="angle of each anchor (default: evenly spaced from 0)")
    ap.add_argument("--out", default=str(HERE / "rotation_output"))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--steps", type=int, default=360, help="frames per full circle")
    ap.add_argument("--mode", choices=["circular", "linear"], default="circular")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--oval", type=int, nargs=4, default=[89, 128, 42, 58],
                    metavar=("CX", "CY", "AX", "AY"))
    ap.add_argument("--feather", type=float, default=9.0)
    ap.add_argument("--bg", type=int, default=128)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    n = len(args.images)
    angles = np.array(args.angles if args.angles else [360 * i / n for i in range(n)], float)
    assert len(angles) == n and np.all(np.diff(angles) > 0) and angles[-1] - angles[0] < 360
    gaps = np.diff(np.append(angles, angles[0] + 360))
    assert np.all(gaps < 180), "neighbouring anchors must be < 180 deg apart"

    imgs = [np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in args.images]
    h, w = imgs[0].shape[:2]
    border = border_points(h, w)
    shapes = [np.vstack([detect_landmarks(im.astype(np.uint8), args.model), border])
              for im in imgs]

    mean_shape = np.mean(shapes, axis=0)
    triangles = Delaunay(mean_shape).simplices
    # Mean texture, in the mean shape.
    mean_img = np.mean([warp_to(im, s, mean_shape, triangles) for im, s in zip(imgs, shapes)],
                       axis=0)

    mask = None if args.no_mask else oval_mask(h, w, tuple(args.oval[:2]),
                                               tuple(args.oval[2:]), args.feather)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    rows = []
    for step in range(args.steps):
        theta = angles[0] + 360 * step / args.steps
        k = int(np.searchsorted(angles, (theta - angles[0]) % 360 + angles[0], side="right") - 1)
        k2 = (k + 1) % n
        phi = (theta - angles[k]) % 360
        w1, w2, wm = weights(phi, gaps[k], args.mode)

        shape = w1 * shapes[k] + w2 * shapes[k2] + wm * mean_shape
        frame = w1 * warp_to(imgs[k], shapes[k], shape, triangles) \
            + w2 * warp_to(imgs[k2], shapes[k2], shape, triangles)
        if wm != 0:
            frame += wm * warp_to(mean_img, mean_shape, shape, triangles)
        if mask is not None:
            frame = frame * mask + args.bg * (1 - mask)
        Image.fromarray(np.clip(frame, 0, 255).round().astype(np.uint8)).save(
            out / "frames" / f"frame_{step:04d}.png")
        rows.append({"step": step, "angle": theta % 360, "anchor_a": k, "anchor_b": k2,
                     "w_a": w1, "w_b": w2, "w_mean": wm})

    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)
    Image.fromarray(np.clip(mean_img, 0, 255).astype(np.uint8)).save(out / "mean_face.png")

    if args.gif:
        frames = [Image.open(f) for f in sorted((out / "frames").glob("frame_*.png"))]
        frames[0].save(out / "rotation.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
    print(f"Saved {args.steps} frames to {out}")


if __name__ == "__main__":
    main()
