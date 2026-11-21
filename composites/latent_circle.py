#################################################################
# latent_circle.py
#
# A closed circle of generated faces through two women A and B, in the
# StyleGAN W space - the same geometry as the rotation in the project's
# PCA space (generate_rotation_sequence.py):
#
#     w(theta) = c + cos(theta) * a + sin(theta) * b
#
#   c = midpoint of A and B,  a = w_A - c  (so theta = 0 is A, 180 is B)
#   b = direction of a third woman from c, made orthogonal to a and scaled
#       to |a| (so the circle is round).
#
# theta 0 -> 180 goes from A to B through c + b (90 deg); theta 180 -> 360
# comes back from B to A through c - b (270 deg), so the way back is made
# of different faces.
#
# Every face has the same frontal pose (W layers 0-2 = average face) and
# the same colour (layers 8-17 = mean of A and B), as in
# noise_average_generated.py, so the faces differ only in features.
#
# Output: <out>/<A>_to_<B>/ (180 frames, theta 0..179),
#         <out>/<B>_to_<A>/ (180 frames, theta 180..359),
#         circle.csv (angle, identity similarity to A and B), circle.gif,
#         circle.mp4, strip.png
#
#   py composites/latent_circle.py --seeds 38 46 --out <folder>/circle
#################################################################
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from generate_faces import load_generator, parse_seeds, seed_to_w, synth
from pick_different_pair import DEFAULT_SEEDS, embed, load_resnet

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[38, 46], help="A (0 deg) and B (180 deg)")
    ap.add_argument("--third", type=int, default=None,
                    help="seed that sets the 90/270 deg direction (default: the pool woman "
                         "most orthogonal to the A-B axis)")
    ap.add_argument("--pool", default=DEFAULT_SEEDS, help="candidates for --third")
    ap.add_argument("--steps", type=int, default=360, help="frames on the full circle")
    ap.add_argument("--psi", type=float, default=0.7)
    ap.add_argument("--pose-layers", type=int, default=3)
    ap.add_argument("--color-from", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(device)
    w_avg = G.mapping.w_avg[None, None, :]

    def base(seed):
        w = seed_to_w(G, seed, args.psi, device)
        w[:, :args.pose_layers] = w_avg
        return w

    wa, wb = base(args.seeds[0]), base(args.seeds[1])
    c = (wa + wb) / 2
    c_col = c[:, args.color_from:].clone()

    def fix_colour(w):
        w = w.clone()
        w[:, args.color_from:] = c_col
        return w

    a = fix_colour(wa) - fix_colour(c)
    a_flat = a.flatten()

    # Third direction: orthogonal part of (third woman - c), same length as a.
    cands = [args.third] if args.third is not None else [
        s for s in parse_seeds(args.pool) if s not in args.seeds]
    best = None
    for s in cands:
        d = (fix_colour(base(s)) - fix_colour(c)).flatten()
        d_perp = d - (d @ a_flat) / (a_flat @ a_flat) * a_flat
        ratio = float(d_perp.norm() / d.norm())
        if best is None or ratio > best[0]:
            best = (ratio, s, d_perp)
    ratio, third, d_perp = best
    b = (d_perp / d_perp.norm() * a_flat.norm()).view_as(a)
    print(f"90/270 deg direction from seed {third} ({ratio:.0%} of it orthogonal to A-B)")

    out = Path(args.out)
    na, nb = args.seeds
    half = args.steps // 2
    dirs = [out / f"{na}_to_{nb}", out / f"{nb}_to_{na}"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    net = load_resnet(device)
    frames, rows = [], []
    ref = {}
    for step in range(args.steps):
        theta = 360.0 * step / args.steps
        t = np.deg2rad(theta)
        w = fix_colour(c) + float(np.cos(t)) * a + float(np.sin(t)) * b
        img = cv2.resize(synth(G, w), (args.size, args.size), interpolation=cv2.INTER_AREA)
        folder = dirs[0] if step < half else dirs[1]
        name = f"circle_{theta:05.1f}deg.png"
        Image.fromarray(img).save(folder / name)
        e = embed(net, img, device)
        if step == 0:
            ref["A"] = e
        if step == half:
            ref["B"] = e
        frames.append(img)
        rows.append({"step": step, "angle": theta, "folder": folder.name, "file": name, "emb": e})

    for r in rows:
        r["sim_A"] = round(float(r["emb"] @ ref["A"]), 4)
        r["sim_B"] = round(float(r["emb"] @ ref["B"]), 4)
        del r["emb"]
    df = pd.DataFrame(rows)
    df.to_csv(out / "circle.csv", index=False)

    # Strip every 30 deg, GIF and MP4 of the full loop.
    show = list(range(0, args.steps, args.steps // 12))
    th = 192
    strip = Image.new("RGB", (len(show) * th, th + 18), "white")
    d = ImageDraw.Draw(strip)
    for k, i in enumerate(show):
        strip.paste(Image.fromarray(frames[i]).resize((th, th), Image.LANCZOS), (k * th, 18))
        d.text((k * th + 4, 3), f"{rows[i]['angle']:.0f} deg", fill="black")
    strip.save(out / "strip.png")
    small = [Image.fromarray(f).resize((256, 256), Image.LANCZOS) for f in frames]
    small[0].save(out / "circle.gif", save_all=True, append_images=small[1:], duration=40, loop=0)
    wr = imageio.get_writer(out / "circle.mp4", fps=30, codec="libx264", quality=9,
                            pixelformat="yuv420p", macro_block_size=16)
    for f in frames:
        wr.append_data(f)
    wr.close()

    for q in (0, 45, 90, 135, 180, 225, 270, 315):
        r = df.iloc[int(q * args.steps / 360)]
        print(f"  {q:3d} deg: similarity to {na} = {r.sim_A:.2f}, to {nb} = {r.sim_B:.2f}")
    print(f"Saved {args.steps} frames to {out}")


if __name__ == "__main__":
    main()
