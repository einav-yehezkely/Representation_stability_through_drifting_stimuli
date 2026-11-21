#################################################################
# make_bigfive_style.py
#
# Composites in the style of the classic personality-trait composites
# (e.g. the Big Five face averages): shape-normalised averages.
#
# For each angle (default 0 and 180) on the PCA rotation path:
# 1. Take the faces nearest to the target point (after identity dedup).
# 2. Run MediaPipe FaceLandmarker (478 points + blendshapes) on each and
#    keep only neutral, frontal faces: no smile, mouth closed, head not turned.
# 3. Warp every kept face to the group's mean shape (Delaunay piecewise
#    affine) and average the pixels. Eyes, nose and mouth line up exactly,
#    so they stay sharp; hair and background blur into a soft halo.
# 4. No mask; a square crop around the face, upscaled.
#
# Landmarks are cached in composites/cache/full_landmarks.npz.
#
#   py composites/make_bigfive_style.py --angles 0 180 --k 100
#################################################################
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

from make_rotation_composites import dedup_identities, load_embeddings, load_points
from morph_two import DEFAULT_MODEL, border_points, warp_to

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "composites" / "cache" / "full_landmarks.npz"
BLENDSHAPES = ["mouthSmileLeft", "mouthSmileRight", "jawOpen"]


def landmarks_for(names, image_dir, upscale=2):
    """Returns {name: (pts (478,2) or None, smile, jaw_open)}, cached."""
    known = {}
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        known = dict(zip(d["names"], zip(d["pts"], d["smile"], d["jaw"])))
    missing = [n for n in names if n not in known]
    if missing:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.3, output_face_blendshapes=True)
        with vision.FaceLandmarker.create_from_options(options) as det:
            for i, n in enumerate(missing):
                img = np.asarray(Image.open(Path(image_dir) / n).convert("RGB"))
                h, w = img.shape[:2]
                big = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
                r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=np.ascontiguousarray(big)))
                if not r.face_landmarks:
                    known[n] = (np.full((478, 2), np.nan, np.float32), np.nan, np.nan)
                    continue
                pts = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32)
                bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
                smile = (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2
                known[n] = (pts, smile, bs["jawOpen"])
                if (i + 1) % 200 == 0:
                    print(f"  landmarks {i + 1}/{len(missing)}")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        keys = list(known)
        np.savez(CACHE, names=np.array(keys), pts=np.stack([known[k][0] for k in keys]),
                 smile=np.array([known[k][1] for k in keys]),
                 jaw=np.array([known[k][2] for k in keys]))
    return {n: known[n] for n in names}


def yaw_ratio(pts):
    """Nose-tip offset from the eye midpoint, relative to eye distance
    (0 = frontal)."""
    le, re = pts[[33, 133]].mean(0), pts[[362, 263]].mean(0)
    return abs(pts[1, 0] - (le[0] + re[0]) / 2) / np.linalg.norm(re - le)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "pca_top2_filtered_female_vgg_1.csv"))
    ap.add_argument("--images", default=str(ROOT / "female_faces"))
    ap.add_argument("--out", default=str(ROOT / "composites" / "bigfive_style"))
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 180])
    ap.add_argument("--radius", type=float, default=0.45)
    ap.add_argument("--k", type=int, default=100, help="faces per composite (after filtering)")
    ap.add_argument("--pool", type=int, default=500, help="nearest faces to screen per angle")
    ap.add_argument("--max-smile", type=float, default=0.25)
    ap.add_argument("--max-jaw", type=float, default=0.15)
    ap.add_argument("--max-yaw", type=float, default=0.12)
    ap.add_argument("--size", type=int, default=512, help="output size in px (square)")
    ap.add_argument("--max-sim", type=float, default=0.75)
    args = ap.parse_args()

    names, points = load_points(args.csv)
    keep = dedup_identities(load_embeddings(names), args.max_sim)
    names, points = names[keep], points[keep]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for ang in args.angles:
        t = np.deg2rad(ang)
        d = np.linalg.norm(points - args.radius * np.array([np.cos(t), np.sin(t)]), axis=1)
        order = np.argsort(d)[:args.pool]
        lm = landmarks_for(list(names[order]), args.images)

        chosen = []
        for i in order:
            pts, smile, jaw = lm[names[i]]
            if np.isnan(pts).any() or smile > args.max_smile or jaw > args.max_jaw \
                    or yaw_ratio(pts) > args.max_yaw:
                continue
            chosen.append(i)
            if len(chosen) == args.k:
                break
        print(f"{ang:.0f} deg: {len(chosen)} neutral frontal faces "
              f"(screened {args.pool}, farthest used at rank {list(order).index(chosen[-1]) + 1})")

        imgs = [np.asarray(Image.open(Path(args.images) / names[i]).convert("RGB"))
                .astype(np.float32) for i in chosen]
        h, w = imgs[0].shape[:2]
        border = border_points(h, w)
        shapes = [np.vstack([lm[names[i]][0], border]) for i in chosen]
        mean_shape = np.mean(shapes, axis=0)
        tri = Delaunay(mean_shape).simplices
        avg = np.mean([warp_to(im, s, mean_shape, tri) for im, s in zip(imgs, shapes)], axis=0)

        # Square crop centred on the face (between the eyes and the mouth).
        cx = mean_shape[[33, 263, 13], 0].mean()
        cy = (mean_shape[[33, 263], 1].mean() + mean_shape[13, 1]) / 2
        half = min(w / 2, cx, w - cx, cy, h - cy)
        crop = avg[int(cy - half):int(cy + half), int(cx - half):int(cx + half)]
        img = Image.fromarray(np.clip(crop, 0, 255).round().astype(np.uint8))
        img = img.resize((args.size, args.size), Image.LANCZOS)
        img.save(out / f"composite_{ang:.0f}.png")
        results.append(np.asarray(img))
        with open(out / f"composite_{ang:.0f}_faces.txt", "w") as f:
            f.write("\n".join(names[chosen]))

    gap = np.full((args.size, 24, 3), 255, np.uint8)
    row = [results[0]]
    for r in results[1:]:
        row += [gap, r]
    Image.fromarray(np.hstack(row)).save(out / "side_by_side.png")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
