#################################################################
# noise_average_generated.py
#
# Start from generated StyleGAN faces (seeds from generate_faces.py), make
# many noisy variants around each one, and average the variants into a
# composite (Big-Five-composite look).
#
# Noise is added in the generator's latent W space, so every variant is
# still a realistic face that resembles the original:
#     w_i = (1 - a) * w_seed + a * w_random_i
# where w_random_i is the latent of a random face and a = --noise.
# The variants are then averaged in pixel space after warping each to their
# mean landmark shape (sharp eyes/mouth, soft hair), as in
# make_bigfive_style.py. A plain (unwarped) average is saved too.
#
# Head pose lives in the coarse W layers, so with --pose-layers N (default 3)
# those layers come from the average face for every image: all faces look
# straight ahead and differ only in facial features.
#
# --exaggerate s pushes the original faces apart (caricature) in W space,
# w = mean + s * (w - mean), so their feature differences are s times larger.
# The pose layers are identical for all seeds, so they are not affected.
#
# --color-from K gives every face and variant the same fine layers K..17
# (the mean of the seeds'), which carry colour: hair, skin and eye colour.
# With K = 8 both sides get the same hair colour and the differences left
# are in the facial features.
#
# --hair-layers 4 --hair-noise 0.8 gives the hairstyle layer much more noise
# than the rest, so the variants vary in hairstyle (e.g. not all with a
# fringe) while the facial features stay close to the original.
#
# --noise-pool SEEDS draws the random face only from a pool of chosen seeds
# (e.g. adult women), as a random convex mix of --pool-mix of them, so no
# variant drifts toward a man or a child.
#
# --edit SEED:DONOR:LAYERS copies some W layers of another seed into one
# face, e.g. 38:83:3,4 replaces the hairstyle layers of seed 38 with those
# of seed 83, which removes 38's fringe.
#
#   py composites/noise_average_generated.py --seeds 0 83 --n 100 --noise 0.5
#################################################################
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import Delaunay

from generate_faces import load_generator, seed_to_w, synth
from morph_two import DEFAULT_MODEL, border_points, warp_to

HERE = Path(__file__).resolve().parent


def detector():
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(DEFAULT_MODEL)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3)
    return vision.FaceLandmarker.create_from_options(options)


def landmarks(det, img):
    import mediapipe as mp

    h, w = img.shape[:2]
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not r.face_landmarks:
        return None
    return np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]], np.float32)


def sheet(imgs, cols, thumb):
    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, im in enumerate(imgs):
        out.paste(Image.fromarray(im).resize((thumb, thumb), Image.LANCZOS),
                  ((i % cols) * thumb, (i // cols) * thumb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 83])
    ap.add_argument("--n", type=int, default=100, help="noisy variants per face")
    ap.add_argument("--noise", type=float, default=0.5,
                    help="how far each variant moves toward a random face (0-1)")
    ap.add_argument("--psi", type=float, default=0.7, help="truncation of the original faces")
    ap.add_argument("--size", type=int, default=512, help="working/output size in px")
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--pose-layers", type=int, default=3,
                    help="replace the first N (coarse) W layers of every image with the "
                         "average face's, so all faces share the same frontal head pose and "
                         "differ only in facial features (0 = off)")
    ap.add_argument("--exaggerate", type=float, default=1.0,
                    help="push the faces apart in W space before adding noise: "
                         "w = mean + s * (w - mean) over the given seeds (1 = off)")
    ap.add_argument("--color-from", type=int, default=0,
                    help="share W layers from this index on (fine layers = hair, skin and eye "
                         "colour) between all faces and variants: they get the mean of the "
                         "seeds' layers. 8 gives the same hair colour while keeping features "
                         "(0 = off)")
    ap.add_argument("--hair-layers", type=int, nargs="*", default=[],
                    help="W layers that get their own (stronger) noise, e.g. 4 = hairstyle "
                         "and fringe")
    ap.add_argument("--hair-noise", type=float, default=None,
                    help="noise level for --hair-layers (default: same as --noise)")
    ap.add_argument("--noise-pool", default=None,
                    help="draw the random face only from these seeds (e.g. adult women), as a "
                         "random convex mix of --pool-mix of them; default: any random face")
    ap.add_argument("--pool-mix", type=int, default=3)
    ap.add_argument("--edit", action="append", default=[], metavar="SEED:DONOR:LAYERS",
                    help="give SEED the W layers LAYERS (comma-separated) of seed DONOR, e.g. "
                         "38:83:3,4 removes the fringe of seed 38 (hairstyle layers). "
                         "Can be repeated")
    ap.add_argument("--out", default=None,
                    help="default: composites/generated/noise_average/"
                         "noise_<level>[_pose<N>][_ex<s>][_col<K>]")
    args = ap.parse_args()
    if args.out is None:
        name = f"noise_{args.noise:.2f}" + (f"_pose{args.pose_layers}" if args.pose_layers else "")
        name += f"_ex{args.exaggerate:g}" if args.exaggerate != 1 else ""
        name += f"_col{args.color_from}" if args.color_from else ""
        args.out = str(HERE / "generated" / "noise_average" / name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.noise_seed)
    det = detector()
    w_pose = G.mapping.w_avg[None, None, :]  # the average face: frontal, upright
    pool = None
    if args.noise_pool:
        from generate_faces import parse_seeds
        pool = [seed_to_w(G, s, args.psi, device) for s in parse_seeds(args.noise_pool)
                if s not in args.seeds]
        print(f"Noise faces drawn from {len(pool)} pool seeds")

    # Latents of the original faces, pose fixed, optionally pushed apart.
    ws = []
    for seed in args.seeds:
        w = seed_to_w(G, seed, args.psi, device)
        if args.pose_layers:
            w[:, :args.pose_layers] = w_pose
        ws.append(w)
    for e in args.edit:
        seed, donor, layers = e.split(":")
        layers = [int(x) for x in layers.split(",")]
        w_d = seed_to_w(G, int(donor), args.psi, device)
        ws[args.seeds.index(int(seed))][:, layers] = w_d[:, layers]
    center = torch.stack(ws).mean(0)
    if args.exaggerate != 1:
        ws = [center + args.exaggerate * (w - center) for w in ws]
    if args.color_from:
        for w in ws:
            w[:, args.color_from:] = center[:, args.color_from:]

    aligned_all, plain_all, originals = [], [], []
    for seed, w0 in zip(args.seeds, ws):
        folder = out / f"seed{seed:04d}"
        folder.mkdir(exist_ok=True)
        orig = cv2.resize(synth(G, w0), (args.size, args.size), interpolation=cv2.INTER_AREA)
        originals.append(orig)

        imgs, shapes = [], []
        with torch.no_grad():
            for i in range(args.n):
                if pool is None:
                    z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
                    w_rand = G.mapping(z, None, truncation_psi=1.0)
                else:
                    pick = rng.choice(len(pool), args.pool_mix, replace=False)
                    alpha = rng.dirichlet(np.ones(args.pool_mix))
                    w_rand = sum(float(a) * pool[k] for a, k in zip(alpha, pick))
                w = (1 - args.noise) * w0 + args.noise * w_rand
                if args.hair_layers and args.hair_noise is not None:
                    h = args.hair_layers
                    w[:, h] = (1 - args.hair_noise) * w0[:, h] + args.hair_noise * w_rand[:, h]
                if args.pose_layers:
                    w[:, :args.pose_layers] = w_pose
                if args.color_from:
                    w[:, args.color_from:] = center[:, args.color_from:]
                img = cv2.resize(synth(G, w), (args.size, args.size), interpolation=cv2.INTER_AREA)
                Image.fromarray(img).save(folder / f"seed{seed:04d}_variant_{i:03d}.png")
                pts = landmarks(det, img)
                if pts is None:
                    continue
                imgs.append(img.astype(np.float32))
                shapes.append(pts)
        print(f"seed {seed}: {args.n} variants, landmarks found in {len(imgs)}")

        sheet([orig] + [im.astype(np.uint8) for im in imgs[:23]], 8, 160).save(
            out / f"variants_seed{seed:04d}.png")

        border = border_points(args.size, args.size)
        shapes = [np.vstack([s, border]) for s in shapes]
        mean_shape = np.mean(shapes, axis=0)
        tri = Delaunay(mean_shape).simplices
        aligned = np.mean([warp_to(im, s, mean_shape, tri) for im, s in zip(imgs, shapes)], axis=0)
        plain = np.mean(imgs, axis=0)
        for name, img in [("average", aligned), ("average_plain", plain)]:
            Image.fromarray(np.clip(img, 0, 255).round().astype(np.uint8)).save(
                out / f"{name}_seed{seed:04d}.png")
        aligned_all.append(np.clip(aligned, 0, 255).round().astype(np.uint8))
        plain_all.append(np.clip(plain, 0, 255).round().astype(np.uint8))

    # Rows: original, plain average, aligned average; one column per seed.
    gap_v = np.full((args.size, 24, 3), 255, np.uint8)
    rows = []
    for group in (originals, plain_all, aligned_all):
        row = [group[0]]
        for g in group[1:]:
            row += [gap_v, g]
        rows.append(np.hstack(row))
    gap_h = np.full((24, rows[0].shape[1], 3), 255, np.uint8)
    grid = np.vstack([rows[0], gap_h, rows[1], gap_h, rows[2]])
    Image.fromarray(grid).save(out / "summary.png")
    Image.fromarray(np.hstack([aligned_all[0], gap_v, aligned_all[-1]])).save(
        out / "side_by_side.png")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
