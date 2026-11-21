#################################################################
# make_noise_stimuli.py
#
# Two opposite composite faces, A (angle `start`) and B (start + 180),
# each degraded by a varying amount of "noise", where the noise is a third
# composite face N at start + 90 (orthogonal to both in the PCA space).
#
# For each side X in {A, B} and noise level a in [0, max_noise]:
#     stimulus = (1 - a) * X + a * N
# as a landmark morph (shape and texture), exactly like morph_endpoints.py.
# a = 0 is the clean face; as a -> 1 both sides converge to the same face N,
# so A and B become harder to tell apart.
#
# Output (default composites/noise_stimuli/):
#   A_0/noise_00.png ... noise_49.png, B_180/..., N_90.png, A_0.png, B_180.png,
#   stimuli.csv (side, angle, level, noise, file), overview.png, mask.png
#
#   py composites/make_noise_stimuli.py --n 50
#################################################################
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from morph_endpoints import add_common_args, blend, prepare_composites


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap, "noise_stimuli")
    ap.add_argument("--start", type=float, default=0.0, help="angle of A; B at +180, noise at +90")
    ap.add_argument("--noise-angle", type=float, default=None,
                    help="angle of the noise face (default start + 90)")
    ap.add_argument("--n", type=int, default=50, help="images per side")
    ap.add_argument("--max-noise", type=float, default=0.98,
                    help="noise weight of the last image (levels are evenly spaced from 0)")
    args = ap.parse_args()

    a_ang = args.start % 360
    b_ang = (args.start + 180) % 360
    n_ang = (args.start + 90) % 360 if args.noise_angle is None else args.noise_angle % 360

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = prepare_composites([a_ang, b_ang, n_ang], args, out)
    ends, pts, triangles, mask = c["ends"], c["pts"], c["triangles"], c["mask"]
    noise_img, noise_pts = ends[2], pts[2]

    levels = np.linspace(0, args.max_noise, args.n)
    rows = []
    for side, ang, img, p in [("A", a_ang, ends[0], pts[0]), ("B", b_ang, ends[1], pts[1])]:
        folder = out / f"{side}_{ang:.0f}"
        folder.mkdir(exist_ok=True)
        for i, a in enumerate(levels):
            f = folder / f"noise_{i:02d}.png"
            blend(img, p, noise_img, noise_pts, a, triangles, mask, args.bg).save(f)
            rows.append({"side": side, "angle": ang, "level": i, "noise": round(float(a), 4),
                         "file": f.relative_to(out).as_posix()})
        Image.open(folder / "noise_00.png").save(out / f"{side}_{ang:.0f}.png")
    blend(noise_img, noise_pts, noise_img, noise_pts, 0, triangles, mask, args.bg).save(
        out / f"N_{n_ang:.0f}.png")
    pd.DataFrame(rows).to_csv(out / "stimuli.csv", index=False)

    # Overview: A row and B row, every ~7th level, cropped to the face.
    ys, xs = np.nonzero(mask[..., 0] > 0.02)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    show = np.unique(np.linspace(0, args.n - 1, 8).round().astype(int))
    grid = []
    for side, ang in [("A", a_ang), ("B", b_ang)]:
        grid.append(np.hstack([
            np.asarray(Image.open(out / f"{side}_{ang:.0f}" / f"noise_{i:02d}.png"))[y0:y1, x0:x1]
            for i in show]))
    sheet = Image.fromarray(np.vstack(grid))
    sheet.resize((sheet.width * 2, sheet.height * 2), Image.LANCZOS).save(out / "overview.png")

    print(f"Saved {len(rows)} stimuli to {out} (noise levels {', '.join(f'{levels[i]:.2f}' for i in show)} "
          f"shown in overview.png)")


if __name__ == "__main__":
    main()
