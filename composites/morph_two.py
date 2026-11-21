#################################################################
# morph_two.py
#
# Classic landmark-based face morph between two images (A -> B),
# using only the two images themselves (no face database).
#
# 1. Detect 478 MediaPipe face landmarks on A and on B
#    (+ points on the image border so the background warps too).
# 2. Triangulate the mean shape (Delaunay).
# 3. For each t in [0, 1]:
#       shape_t = (1 - t) * shape_A + t * shape_B
#       warp A and B to shape_t (piecewise affine, per triangle)
#       frame_t = (1 - t) * A_warped + t * B_warped
# 4. Optionally apply the inner-face oval mask (hides hair).
#
# Frame i is labelled with angle = start + range * t, so a 0->180 morph
# gives the same file layout as make_rotation_composites.py.
#################################################################
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import Delaunay

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "face_landmarker.task"


def detect_landmarks(img_rgb, model_path, upscale=4):
    """Return (478, 2) landmark pixel coordinates. Small images are upscaled
    before detection, which makes MediaPipe much more reliable."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    h, w = img_rgb.shape[:2]
    big = cv2.resize(img_rgb, (w * upscale, h * upscale), interpolation=cv2.INTER_CUBIC)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
    )
    with vision.FaceLandmarker.create_from_options(options) as detector:
        result = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(big))
        )
    if not result.face_landmarks:
        raise RuntimeError("No face detected")
    return np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], np.float32)


def border_points(h, w):
    xs, ys = [0, w // 2, w - 1], [0, h // 2, h - 1]
    return np.array([[x, y] for y in ys for x in xs if not (x == w // 2 and y == h // 2)],
                    np.float32)


def warp_to(img, src_pts, dst_pts, triangles):
    """Piecewise-affine warp of img so that src_pts land on dst_pts."""
    out = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape[:2]
    src_pts = np.asarray(src_pts, np.float32)
    dst_pts = np.asarray(dst_pts, np.float32)
    for tri in triangles:
        s, d = src_pts[tri], dst_pts[tri]
        x, y, bw, bh = cv2.boundingRect(d)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + bw, w), min(y + bh, h)
        if x1 <= x0 or y1 <= y0:
            continue
        M = cv2.getAffineTransform(s, d - np.float32([x0, y0]))
        patch = cv2.warpAffine(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillConvexPoly(mask, np.int32(np.round(d - [x0, y0])), 1.0, cv2.LINE_AA)
        region = out[y0:y1, x0:x1]
        region[:] = region * (1 - mask[..., None]) + patch * mask[..., None]
    return out


def oval_mask(h, w, center, axes, feather):
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_a")
    ap.add_argument("image_b")
    ap.add_argument("--out", default=str(HERE / "morph_output"))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--start", type=float, default=0.0, help="angle label of image A")
    ap.add_argument("--range", type=float, default=180.0, help="angle span A -> B")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--oval", type=int, nargs=4, default=[89, 128, 42, 58],
                    metavar=("CX", "CY", "AX", "AY"))
    ap.add_argument("--feather", type=float, default=9.0)
    ap.add_argument("--bg", type=int, default=128)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    a = np.asarray(Image.open(args.image_a).convert("RGB"))
    b = np.asarray(Image.open(args.image_b).convert("RGB"))
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]

    border = border_points(h, w)
    pts_a = np.vstack([detect_landmarks(a, args.model), border])
    pts_b = np.vstack([detect_landmarks(b, args.model), border])
    triangles = Delaunay((pts_a + pts_b) / 2).simplices

    mask = None if args.no_mask else oval_mask(h, w, tuple(args.oval[:2]),
                                               tuple(args.oval[2:]), args.feather)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    a_f, b_f = a.astype(np.float32), b.astype(np.float32)
    rows = []
    for step in range(args.steps + 1):
        t = step / args.steps
        pts_t = (1 - t) * pts_a + t * pts_b
        frame = (1 - t) * warp_to(a_f, pts_a, pts_t, triangles) \
            + t * warp_to(b_f, pts_b, pts_t, triangles)
        if mask is not None:
            frame = frame * mask + args.bg * (1 - mask)
        Image.fromarray(np.clip(frame, 0, 255).round().astype(np.uint8)).save(
            out / "frames" / f"frame_{step:04d}.png")
        rows.append({"step": step, "t": t, "angle_label": (args.start + args.range * t) % 360})

    pd.DataFrame(rows).to_csv(out / "frames.csv", index=False)

    # Landmark overlay, to check the detection by eye.
    check = np.hstack([a.copy(), b.copy()])
    for p in pts_a[:-len(border)]:
        cv2.circle(check, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)
    for p in pts_b[:-len(border)]:
        cv2.circle(check, (int(p[0]) + w, int(p[1])), 1, (0, 255, 0), -1)
    Image.fromarray(check).save(out / "landmarks_check.png")

    if args.gif:
        frames = [Image.open(f) for f in sorted((out / "frames").glob("frame_*.png"))]
        frames += frames[-2:0:-1]
        frames[0].save(out / "morph.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0)
    print(f"Saved {args.steps + 1} frames to {out}")


if __name__ == "__main__":
    main()
