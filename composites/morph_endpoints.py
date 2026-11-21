#################################################################
# morph_endpoints.py
#
# Morph between only TWO composite faces: the one at angle `start` (A) and
# the one at the opposite angle `start + 180` (B) on the PCA rotation path.
# Nothing is averaged for the in-between frames; they come from the two
# endpoint images alone.
#
# 1. Build A and B as in make_rotation_composites.py (weighted average of
#    the k nearest faces, identity dedup), without a mask, so the whole
#    head is visible for landmark detection.
# 2. Process them: optional unsharp mask (composites are blurry), and the
#    same low-frequency equalization as the rotation, so A and B get the
#    same edge light/colour pattern (no dark-vs-blonde halo).
# 3. Landmark morph (MediaPipe 478 points + Delaunay warp), as in
#    morph_two.py: shape and texture go linearly from A to B.
# 4. One fixed face-contour mask (from the mean of A and B) on every frame.
#
# Output (default composites/endpoint_morph/): A_<start>.png, B_<end>.png,
# frames/, frames.csv, morph.gif, landmarks_check.png, mask.png.
#
#   py composites/morph_endpoints.py --steps 90 --gif
#################################################################
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import Delaunay

from make_rotation_composites import (ImageCache, composite_at, contour_mask, dedup_identities,
                                      equalize_lowfreq, load_embeddings, load_points,
                                      masked_lowpass)
from morph_two import DEFAULT_MODEL, border_points, detect_landmarks, warp_to

ROOT = Path(__file__).resolve().parent.parent


def sharpen(img, amount, sigma=1.5):
    """Unsharp mask: img + amount * (img - blur(img))."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return img + amount * (img - blur)


def add_common_args(ap, out_name):
    ap.add_argument("--csv", default=str(ROOT / "pca_top2_filtered_female_vgg_1.csv"))
    ap.add_argument("--images", default=str(ROOT / "female_faces"))
    ap.add_argument("--out", default=str(ROOT / "composites" / out_name))
    ap.add_argument("--radius", type=float, default=0.45)
    ap.add_argument("--k", type=int, default=120)
    ap.add_argument("--max-sim", type=float, default=0.75)
    ap.add_argument("--sharpen", type=float, default=0.6,
                    help="unsharp-mask strength for the composites (0 = off)")
    ap.add_argument("--equalize", type=float, default=5.0,
                    help="low-frequency equalization sigma in px (0 = off)")
    ap.add_argument("--shrink", type=float, default=0.97)
    ap.add_argument("--feather", type=float, default=3.0)
    ap.add_argument("--bg", type=int, default=128)
    ap.add_argument("--gif", action="store_true")


def prepare_composites(angles, args, out):
    """Build, landmark and process the composites at the given angles.
    Returns dict with raw/processed images, landmarks (+ border points),
    triangles, the fixed mask and the number of border points."""
    names, points = load_points(args.csv)
    keep = dedup_identities(load_embeddings(names), args.max_sim)
    names, points = names[keep], points[keep]
    cache = ImageCache(args.images)

    raw = [composite_at(a, points, names, cache, args.radius, args.k, args.bg)[0]
           .astype(np.float32) for a in angles]

    # Landmarks on the unmasked composites.
    h, w = raw[0].shape[:2]
    border = border_points(h, w)
    pts = [np.vstack([detect_landmarks(im.astype(np.uint8), DEFAULT_MODEL), border])
           for im in raw]
    mean_shape = np.mean(pts, axis=0)
    triangles = Delaunay(mean_shape).simplices

    # One fixed mask, from all composites warped to their mean shape.
    mean_img = np.mean([warp_to(im, p, mean_shape, triangles) for im, p in zip(raw, pts)], axis=0)
    mask = contour_mask(np.clip(mean_img, 0, 255).astype(np.uint8), args.feather, args.shrink)
    Image.fromarray((mask[..., 0] * 255).round().astype(np.uint8)).save(out / "mask.png")

    # Process: sharpen, then equalize the low frequencies of each composite
    # to their common mean (computed in the mean shape, then warped back).
    ends = [sharpen(im, args.sharpen) for im in raw]
    if args.equalize > 0:
        core = mask[..., 0] > 0.9
        ref_low = masked_lowpass(mean_img, mask, args.equalize)
        processed = []
        for im, p in zip(ends, pts):
            in_mean = warp_to(im, p, mean_shape, triangles)
            eq = equalize_lowfreq(in_mean, mask, args.equalize, ref_low, core)
            delta = warp_to(eq - in_mean, mean_shape, p, triangles)  # correction, own shape
            processed.append(im + delta)
        ends = processed

    # Landmark overlay on the raw composites, to check detection by eye.
    check = np.hstack([np.clip(im, 0, 255).astype(np.uint8) for im in raw]).copy()
    for j, p in enumerate(pts):
        for x, y in p[:-len(border)]:
            cv2.circle(check, (int(x) + j * w, int(y)), 1, (0, 255, 0), -1)
    Image.fromarray(check).save(out / "landmarks_check.png")

    return dict(raw=raw, ends=ends, pts=pts, triangles=triangles, mask=mask)


def blend(img1, pts1, img2, pts2, t, triangles, mask, bg):
    """Landmark morph: (1 - t) * img1 + t * img2 in shape and texture, masked."""
    shape = (1 - t) * pts1 + t * pts2
    frame = (1 - t) * warp_to(img1, pts1, shape, triangles) \
        + t * warp_to(img2, pts2, shape, triangles)
    frame = frame * mask + bg * (1 - mask)
    return Image.fromarray(np.clip(frame, 0, 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "endpoint_morph")
    ap.add_argument("--start", type=float, default=0.0, help="angle of A; B is at start + 180")
    ap.add_argument("--steps", type=int, default=90, help="morph steps from A to B")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    angles = [args.start % 360, (args.start + 180) % 360]
    c = prepare_composites(angles, args, out)
    ends, pts, triangles, mask = c["ends"], c["pts"], c["triangles"], c["mask"]

    rows = []
    for step in range(args.steps + 1):
        t = step / args.steps
        blend(ends[0], pts[0], ends[1], pts[1], t, triangles, mask, args.bg).save(
            out / "frames" / f"frame_{step:04d}.png")
        rows.append({"step": step, "t": t, "target_angle": angles[0] + 180 * t})
    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)

    Image.open(out / "frames" / "frame_0000.png").save(out / f"A_{angles[0]:.0f}.png")
    Image.open(out / "frames" / f"frame_{args.steps:04d}.png").save(out / f"B_{angles[1]:.0f}.png")

    if args.gif:
        frames = [Image.open(out / "frames" / f"frame_{i:04d}.png") for i in range(args.steps + 1)]
        frames += frames[-2:0:-1]
        frames[0].save(out / "morph.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
    print(f"Saved {args.steps + 1} frames to {out}")


if __name__ == "__main__":
    main()
