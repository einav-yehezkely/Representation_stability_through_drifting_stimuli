#################################################################
# make_rotation_composites.py
#
# Builds "composite" (averaged) faces along a rotation in the 2D PCA space.
#
# For every angle theta, the target point is
#     c(theta) = radius * (cos(theta), sin(theta))
# and the frame is a weighted pixel average of the k faces whose PC1/PC2
# coordinates are closest to c(theta). The weight falls smoothly to zero
# at the k-th face, so consecutive frames change gradually. Repeated photos of the same woman are removed once, globally.
#
# theta = 0 and theta = 180 give two opposite composites (A and B);
# sweeping theta gives a smooth rotation-morph between them.
#
# CelebA "img_align_celeba" images are already aligned by the eyes, so a
# plain pixel average already gives a usable composite.
#################################################################
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def load_points(csv_path):
    df = pd.read_csv(csv_path, header=None)
    return df.iloc[:, 0].values, df.iloc[:, [1, 2]].values.astype(float)


def load_embeddings(names):
    """L2-normalised ResNet50-VGGFace2 embeddings of the filtered faces, cached
    as .npy (built once from the full 4GB embeddings CSV)."""
    cache_dir = ROOT / "composites" / "cache"
    npy, names_csv = cache_dir / "emb_resnet50.npy", cache_dir / "emb_names.csv"
    if not npy.exists():
        print("Caching embeddings (one-time)...")
        wanted = set(names)
        out_n, out_v = [], []
        with open(ROOT / "female_resnet50_vggface2_embeddings.csv", encoding="utf-8") as f:
            for line in f:
                n, rest = line.split(",", 1)
                if n in wanted:
                    out_n.append(n)
                    out_v.append(np.array(rest.split(","), dtype=np.float32))
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy, np.stack(out_v))
        pd.Series(out_n).to_csv(names_csv, index=False, header=False)
    emb = np.load(npy)
    emb_names = pd.read_csv(names_csv, header=None)[0].values
    pos = {n: i for i, n in enumerate(emb_names)}
    emb = emb[[pos[n] for n in names]]
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def oval_mask(height, width, center, axes, feather):
    """Soft elliptical mask of the inner face (eyebrows to chin, cheek to cheek).
    CelebA aligned images put the eyes at a fixed position, so one fixed oval
    hides the hair, ears and background of every face."""
    import cv2

    mask = np.zeros((height, width), np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask[..., None]


# MediaPipe face-oval landmark indices (forehead -> jaw -> forehead).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109]


def contour_mask(reference_img, feather, shrink):
    """Soft mask shaped like the face outline (forehead, cheeks, jaw), taken
    from the MediaPipe face-oval landmarks of `reference_img`. `shrink` < 1
    pulls the outline slightly inward, away from hair at the temples."""
    import cv2
    from morph_two import DEFAULT_MODEL, detect_landmarks

    pts = detect_landmarks(reference_img, DEFAULT_MODEL)[FACE_OVAL]
    center = pts.mean(axis=0)
    pts = center + (pts - center) * shrink
    mask = np.zeros(reference_img.shape[:2], np.float32)
    cv2.fillPoly(mask, [np.int32(np.round(pts))], 1.0, cv2.LINE_AA)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask[..., None]


def face_contours(names, image_dir, upscale=2):
    """Face-oval landmarks (36 points, pixel coords) of every photo, detected
    once with MediaPipe and cached. Rows are NaN where no face was found."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from morph_two import DEFAULT_MODEL

    cache_file = ROOT / "composites" / "cache" / "face_oval_landmarks.npz"
    known = {}
    if cache_file.exists():
        data = np.load(cache_file)
        known = dict(zip(data["names"], data["points"]))
    missing = [n for n in names if n not in known]
    if missing:
        print(f"Detecting face outlines in {len(missing)} photos (one-time)...")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
        with vision.FaceLandmarker.create_from_options(options) as detector:
            for n in missing:
                img = np.asarray(Image.open(Path(image_dir) / n).convert("RGB"))
                h, w = img.shape[:2]
                big = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=np.ascontiguousarray(big)))
                if res.face_landmarks:
                    lm = res.face_landmarks[0]
                    known[n] = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_OVAL],
                                        np.float32)
                else:
                    known[n] = np.full((len(FACE_OVAL), 2), np.nan, np.float32)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, names=np.array(list(known)),
                            points=np.stack(list(known.values())))
    return np.stack([known[n] for n in names])


def masked_lowpass(img, mask, sigma):
    """Gaussian blur of img using only pixels inside the (soft) mask, so the
    gray background does not bleed into the face edges."""
    import cv2

    num = cv2.GaussianBlur(img * mask, (0, 0), sigma)
    den = cv2.GaussianBlur(mask[..., 0], (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-4)


def equalize_lowfreq(frame, mask, sigma, ref_low, core):
    """Replace the frame's low-frequency light/colour pattern by the reference
    pattern (mean of all frames), keeping the frame's own overall skin colour
    and all its fine detail (eyes, brows, mouth). Removes hair halos and
    shading differences at the face edges."""
    low = masked_lowpass(frame, mask, sigma)
    offset = low[core].mean(axis=0) - ref_low[core].mean(axis=0)
    return frame - low + ref_low + offset


class ImageCache:
    """Loads photos (uint8). With `contours`, each photo also gets its own soft
    face mask, so hair and background of that photo can be left out of the
    average."""

    def __init__(self, image_dir, contours=None, shrink=0.88, feather=2.0):
        self.image_dir = Path(image_dir)
        self.contours = contours
        self.shrink = shrink
        self.feather = feather
        self.cache = {}

    def _mask(self, pts, shape):
        import cv2

        center = pts.mean(axis=0)
        pts = center + (pts - center) * self.shrink
        mask = np.zeros(shape, np.float32)
        cv2.fillPoly(mask, [np.int32(np.round(pts))], 1.0, cv2.LINE_AA)
        if self.feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.feather)
        return mask

    def get(self, i, name):
        """Returns (image uint8 HxWx3, mask float32 HxW or None)."""
        if name not in self.cache:
            img = np.asarray(Image.open(self.image_dir / name).convert("RGB"))
            mask = None
            if self.contours is not None:
                mask = self._mask(self.contours[i], img.shape[:2])
            self.cache[name] = (img, mask)
        return self.cache[name]


def dedup_identities(emb, max_sim):
    """Keep one photo per woman across the whole dataset: going through the
    photos in order, a photo is dropped if its identity embedding has
    cosine > max_sim with a photo already kept. Done once, globally, so the
    set of faces never changes from one angle to the next."""
    n = len(emb)
    keep = np.ones(n, bool)
    block = 2048
    for s in range(0, n, block):
        sims = emb[s:s + block] @ emb.T
        for r in range(sims.shape[0]):
            i = s + r
            if keep[i]:
                dup = np.nonzero(sims[r, i + 1:] > max_sim)[0] + i + 1
                keep[dup] = False
    return keep


def composite_at(theta_deg, points, names, cache, radius, k, bg=128):
    """Weighted pixel average of the faces around the target point.
    Adaptive biweight kernel: R = distance to the k-th nearest face and
    weight = (1 - (d/R)^2)^2. The number of faces is the same at every angle,
    and a face's weight falls smoothly to 0 as it leaves the neighbourhood,
    so faces fade in and out as the angle changes (no flicker)."""
    theta = np.deg2rad(theta_deg)
    center = radius * np.array([np.cos(theta), np.sin(theta)])

    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)[:k]
    bandwidth = dists[idx[-1]]
    weights = (1 - (dists[idx] / bandwidth) ** 2) ** 2
    weights /= weights.sum()

    # With per-photo masks: num = sum(w * m_i * img_i), den = sum(w * m_i), and
    # each pixel is num / den - the average over only the photos in which that
    # pixel is face, so nobody's hair leaks into the composite.
    num = den = None
    for w, i in zip(weights, idx):
        img, m = cache.get(i, names[i])
        wm = np.full(img.shape[:2], w, np.float32) if m is None else w * m
        contrib = img * wm[..., None]
        num = contrib if num is None else num + contrib
        den = wm if den is None else den + wm
    frame = num / np.maximum(den, 1e-6)[..., None]
    frame[den < 1e-3] = bg

    # Weighted mean position of the faces that were used -
    # tells how far the composite really is from the target point.
    used_center = (weights[:, None] * points[idx]).sum(axis=0)
    n_eff = 1.0 / np.sum(weights ** 2)  # effective number of faces
    return np.clip(frame, 0, 255).astype(np.uint8), idx, weights, used_center, n_eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "pca_top2_filtered_female_vgg_1.csv"))
    ap.add_argument("--images", default=str(ROOT / "female_faces"))
    ap.add_argument("--out", default=str(ROOT / "composites" / "output"))
    ap.add_argument("--radius", type=float, default=0.45)
    ap.add_argument("--k", type=int, default=120,
                    help="faces in each neighbourhood (the farthest gets weight 0); larger = blurrier")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--range", type=float, default=180.0)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--max-sim", type=float, default=0.75,
                    help="skip faces with identity-embedding cosine above this to an already chosen face")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--gif", action="store_true", help="also save a GIF of the frames")
    ap.add_argument("--mask", choices=["contour", "oval", "none"], default="contour",
                    help="contour: face outline from the average of all frames (one fixed "
                         "mask for every frame); oval: fixed ellipse; none: keep hair")
    ap.add_argument("--no-mask", action="store_true", help="same as --mask none")
    ap.add_argument("--shrink", type=float, default=None,
                    help="contour mask: pull the outline inward by this factor "
                         "(default 0.97, or 0.88 with --face-masks)")
    ap.add_argument("--oval", type=int, nargs=4, default=[89, 128, 42, 58],
                    metavar=("CX", "CY", "AX", "AY"),
                    help="oval mask: ellipse center and semi-axes in pixels (CelebA 178x218)")
    ap.add_argument("--feather", type=float, default=None,
                    help="mask edge softness in px (default: 3 for contour, 9 for oval)")
    ap.add_argument("--bg", type=int, default=128, help="background gray level")
    ap.add_argument("--face-masks", action="store_true",
                    help="mask each photo by its own face outline before averaging "
                         "(keeps hair out, but makes the faces look rounder)")
    ap.add_argument("--equalize", type=float, default=5.0,
                    help="give every frame the same low-frequency light/colour pattern "
                         "(that of the mean of all frames), with this blur sigma in px; "
                         "removes the hair halo at the edges. 0 = off")
    ap.add_argument("--face-shrink", type=float, default=0.88,
                    help="per-photo mask: pull each face outline inward by this factor")
    ap.add_argument("--face-feather", type=float, default=2.0,
                    help="per-photo mask: edge softness in px")
    args = ap.parse_args()

    names, points = load_points(args.csv)
    if not args.no_dedup:
        keep = dedup_identities(load_embeddings(names), args.max_sim)
        print(f"Identity dedup: kept {keep.sum()} of {len(keep)} photos")
        names, points = names[keep], points[keep]

    contours = None
    if args.face_masks:
        contours = face_contours(names, args.images)
        found = ~np.isnan(contours).any(axis=(1, 2))
        if not found.all():
            print(f"No face outline found in {(~found).sum()} photos - left out")
        names, points, contours = names[found], points[found], contours[found]
    cache = ImageCache(args.images, contours, args.face_shrink, args.face_feather)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    # A full circle ends where it started, so the last frame is not repeated.
    full_circle = abs(args.range) % 360 == 0
    n_frames = args.steps if full_circle else args.steps + 1

    rows = []
    frames = []
    for step in range(n_frames):
        theta = (args.start + args.range * step / args.steps) % 360
        frame, idx, weights, used_center, n_eff = composite_at(
            theta, points, names, cache, args.radius, args.k, args.bg,
        )
        frames.append(frame)
        rows.append({
            "step": step,
            "target_angle": theta,
            "used_angle": np.degrees(np.arctan2(used_center[1], used_center[0])) % 360,
            "used_radius": float(np.linalg.norm(used_center)),
            "n_faces": len(idx),
            "n_effective": round(float(n_eff), 1),
            "faces": ";".join(f"{names[i]}:{w:.4f}" for i, w in zip(idx, weights)),
        })

    # One fixed mask for all frames, so its outline never moves.
    mask_kind = "none" if args.no_mask else args.mask
    mean_frame = np.mean(frames, axis=0).round().astype(np.uint8)
    mask = None
    if mask_kind == "contour":
        feather = 3.0 if args.feather is None else args.feather
        shrink = args.shrink if args.shrink is not None else (
            0.88 if args.face_masks else 0.97)
        mask = contour_mask(mean_frame, feather, shrink)
    elif mask_kind == "oval":
        feather = 9.0 if args.feather is None else args.feather
        cx, cy, ax, ay = args.oval
        mask = oval_mask(mean_frame.shape[0], mean_frame.shape[1], (cx, cy), (ax, ay), feather)
    if mask is not None:
        Image.fromarray((mask[..., 0] * 255).round().astype(np.uint8)).save(out / "mask.png")

    equalize = mask is not None and args.equalize > 0
    if equalize:
        ref_low = masked_lowpass(np.mean(frames, axis=0), mask, args.equalize)
        core = mask[..., 0] > 0.9
    for step, frame in enumerate(frames):
        frame = frame.astype(np.float32)
        if equalize:
            frame = equalize_lowfreq(frame, mask, args.equalize, ref_low, core)
        if mask is not None:
            frame = frame * mask + args.bg * (1 - mask)
        Image.fromarray(np.clip(frame, 0, 255).round().astype(np.uint8)).save(
            out / "frames" / f"frame_{step:04d}.png")

    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)

    # Save the two endpoints (for a full circle: start and start + 180).
    end_step = args.steps // 2 if full_circle else args.steps
    end_angle = rows[end_step]["target_angle"]
    Image.open(out / "frames" / "frame_0000.png").save(out / f"A_{args.start:.0f}.png")
    Image.open(out / "frames" / f"frame_{end_step:04d}.png").save(out / f"B_{end_angle:.0f}.png")
    if args.gif:
        frames = [Image.open(out / "frames" / f"frame_{i:04d}.png") for i in range(n_frames)]
        if not full_circle:
            frames += frames[-2:0:-1]  # ping-pong back to the start
        frames[0].save(out / "rotation.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0)

    print(f"Saved {n_frames} frames to {out}")


if __name__ == "__main__":
    main()
