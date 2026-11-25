#################################################################
# pca_edge_morph.py
#
# Four generated women at the edges of the generated-women PCA space,
# a circular morph between them, and noisy variants around two of them.
#
# 1. PCA (as in extract_embeddings/PCA_2_and_rest.py) of the FaceNet
#    (InceptionResnetV1-VGGFace2) embeddings of women_generated/
#    (women_generated_facenet_vggface2_embeddings.csv). For each angle
#    0 / 90 / 180 / 270 the woman furthest along that direction
#    (largest projection on (cos t, sin t)) is the anchor.
#
# 2. Morph 0 -> 90 -> 180 -> 270 -> 0 in the StyleGAN W space. Between
#    neighbouring anchors (gap D, offset phi) the latent is a spherical
#    interpolation around the mean latent c of the 4 anchors, as in
#    rotate_morph.py:
#        w = c + sin(D - phi)/sin(D) * (w_k - c) + sin(phi)/sin(D) * (w_k+1 - c)
#    so the distance from the centre stays about constant (rotation, not a
#    straight cut through the middle). Every frame is a real generated face.
#
# 3. --n noisy variants around the anchors at --noise-around (default 0
#    and 180), in W space as in noise_average_generated.py:
#        w_i = (1 - a) * w_anchor + a * w_random_i
#    The random face can turn a variant into a man, so each variant is
#    checked with DeepFace (Woman >= --min-woman) and redrawn if it fails.
#
# Every output image is embedded with the same FaceNet preprocessing
# (256 px -> 160 px, [-1, 1]) and projected into the same PCA, so the
# CSVs give its PC1/PC2 position and pca_overview.png shows where the
# morph and the noise clouds land.
#
# The anchors are rendered exactly as in women_generated/ (psi 0.7, no
# pose or colour fixing), so they sit at their PCA positions.
#
#   py composites/pca_edge_morph.py
#   py composites/pca_edge_morph.py --seeds 17 32 34 45     # choose anchors
#################################################################
import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from generate_faces import load_generator, seed_to_w, synth

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_pca(path):
    """Same PCA as PCA_2_and_rest.py: centre, covariance, top-2 eigenvectors."""
    names, X = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            names.append(parts[0])
            X.append([float(v) for v in parts[1:]])
    X = np.array(X)
    mean = X.mean(0)
    vals, vecs = np.linalg.eigh(np.cov(X - mean, rowvar=False))
    vecs = vecs[:, np.argsort(vals)[::-1][:2]]
    return names, (X - mean) @ vecs, mean, vecs


class FaceNet:
    """Embedding exactly as extract_embeddings/facenet_female_vgg.py does it."""

    def __init__(self, device):
        from facenet_pytorch import InceptionResnetV1
        from torchvision import transforms

        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
        self.tf = transforms.Compose([
            transforms.Resize((160, 160)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)])

    @torch.no_grad()
    def __call__(self, img256):
        x = self.tf(img256)[None].to(self.device)
        return self.model(x)[0].cpu().numpy()


def render(G, w):
    """Full 1024 px render and the 256 px version used in women_generated/."""
    big = Image.fromarray(synth(G, w))
    return big, big.resize((256, 256), Image.LANCZOS)


def export_experiment_set(out):
    """The circle-noise variants (or, without them, the morph frames) and the noise pools in one
    folder, plus a CSV (filename, PC1, PC2; no header) in the format
    human_experiment/scripts/prepare-assets.mjs reads."""
    import shutil

    dst = out / "experiment_set"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    if (out / "circle_noise" / "circle_noise.csv").exists():
        m = pd.read_csv(out / "circle_noise" / "circle_noise.csv")
        src = out / "circle_noise" / "images"
    else:
        m = pd.read_csv(out / "morph" / "morph.csv")
        src = out / "morph" / "frames"
    n = pd.read_csv(out / "noise" / "variants.csv")
    rows = []
    for f, x, y in zip(m.file, m.PC1, m.PC2):
        shutil.copy2(src / f, dst / f)
        rows.append((f, x, y))
    for f, x, y in zip(n.file, n.PC1, n.PC2):
        name = Path(f).name
        shutil.copy2(out / "noise" / f, dst / name)
        rows.append((name, x, y))
    pd.DataFrame(rows).to_csv(out / "experiment_set.csv", index=False, header=False)
    print(f"Experiment set: {len(rows)} images -> {dst}, {out / 'experiment_set.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=str(ROOT / "women_generated_facenet_vggface2_embeddings.csv"))
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="anchor seeds, one per angle (default: furthest woman in each direction)")
    ap.add_argument("--steps", type=int, default=360, help="morph frames on the full circle")
    ap.add_argument("--mode", choices=["spherical", "linear"], default="spherical")
    ap.add_argument("--noise-around", type=float, nargs="+", default=[0, 180],
                    help="angles whose anchor gets noisy variants")
    ap.add_argument("--n", type=int, default=100, help="noisy variants per anchor")
    ap.add_argument("--noise", type=float, default=0.3,
                    help="how far each variant moves toward a random face (0-1)")
    ap.add_argument("--noise-seed", type=int, default=12345)
    ap.add_argument("--min-woman", type=float, default=90.0,
                    help="keep a variant only if DeepFace's Woman score is at least this "
                         "(as in archive/classify_sex_generated.py); rejected ones are redrawn "
                         "(0 = keep all)")
    ap.add_argument("--psi", type=float, default=0.7)
    ap.add_argument("--size", type=int, default=512, help="saved image size in px")
    ap.add_argument("--out", default=str(HERE / "generated" / "pca_edges"))
    ap.add_argument("--circle-noise", action=argparse.BooleanOptionalAction, default=True,
                    help="also make one noisy variant around every morph frame; these replace "
                         "the morph frames in experiment_set/")
    ap.add_argument("--export-only", action="store_true",
                    help="only rebuild experiment_set/ and experiment_set.csv from an existing run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        export_experiment_set(out)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. PCA and anchors ------------------------------------------------
    names, P, emb_mean, vecs = load_pca(args.embeddings)
    seed_of = lambda n: int(Path(n).stem.replace("seed", ""))
    if args.seeds:
        assert len(args.seeds) == len(args.angles), "one seed per angle"
        anchors = list(args.seeds)
    else:
        anchors, used = [], set()
        for t in args.angles:
            u = np.array([np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))])
            proj = P @ u
            proj[list(used)] = -np.inf
            i = int(np.argmax(proj))
            used.add(i)
            anchors.append(seed_of(names[i]))
    pos = {seed_of(n): p for n, p in zip(names, P)}
    print("Anchors:")
    for t, s in zip(args.angles, anchors):
        x, y = pos[s]
        print(f"  {t:5.0f} deg: seed {s:4d}  PC1 {x:+.3f}  PC2 {y:+.3f}  "
              f"(angle {np.degrees(np.arctan2(y, x)) % 360:.0f} deg)")

    G = load_generator(device)
    net = FaceNet(device)
    project = lambda img256: (net(img256) - emb_mean) @ vecs

    ws = [seed_to_w(G, s, args.psi, device) for s in anchors]
    rows, thumbs = [], []
    for t, s, w in zip(args.angles, anchors, ws):
        big, small = render(G, w)
        big.resize((args.size, args.size), Image.LANCZOS).save(out / f"anchor_{t:03.0f}deg_seed{s:04d}.png")
        ref = ROOT / "women_generated" / f"seed{s:04d}.png"
        diff = (np.abs(np.asarray(small, float) - np.asarray(Image.open(ref).convert("RGB"), float)).mean()
                if ref.exists() else float("nan"))
        pc = project(small)
        rows.append({"angle": t, "seed": s, "PC1_csv": pos[s][0], "PC2_csv": pos[s][1],
                     "PC1_rendered": pc[0], "PC2_rendered": pc[1], "pixel_diff_vs_women_generated": diff})
        thumbs.append((f"{t:.0f} deg  seed {s}", small))
    pd.DataFrame(rows).round(4).to_csv(out / "anchors.csv", index=False)
    sheet = Image.new("RGB", (256 * len(thumbs), 276), "white")
    d = ImageDraw.Draw(sheet)
    for k, (label, im) in enumerate(thumbs):
        sheet.paste(im, (k * 256, 20))
        d.text((k * 256 + 4, 4), label, fill="black")
    sheet.save(out / "anchors.png")
    print("Re-rendered anchors vs women_generated/ (mean abs pixel diff): "
          + ", ".join(f"{r['pixel_diff_vs_women_generated']:.2f}" for r in rows))

    # ---- 2. circular morph -------------------------------------------------
    c = torch.stack(ws).mean(0)
    ang = np.array(args.angles, float)
    order = np.argsort(ang)
    ang, ws_sorted = ang[order], [ws[i] for i in order]
    mdir = out / "morph" / "frames"
    mdir.mkdir(parents=True, exist_ok=True)
    mrows, frames, mws = [], [], []
    for step in range(args.steps):
        theta = 360.0 * step / args.steps
        k = int(np.searchsorted(ang, theta, side="right") - 1) % len(ang)
        k1 = (k + 1) % len(ang)
        gap = (ang[k1] - ang[k]) % 360 or 360
        phi = (theta - ang[k]) % 360
        if args.mode == "linear":
            a1, a2 = 1 - phi / gap, phi / gap
        else:
            g, p = np.deg2rad(gap), np.deg2rad(phi)
            a1, a2 = np.sin(g - p) / np.sin(g), np.sin(p) / np.sin(g)
        w = c + float(a1) * (ws_sorted[k] - c) + float(a2) * (ws_sorted[k1] - c)
        mws.append(w)
        big, small = render(G, w)
        img = big.resize((args.size, args.size), Image.LANCZOS)
        name = f"morph_{theta:05.1f}deg.png"
        img.save(mdir / name)
        pc = project(small)
        mrows.append({"step": step, "angle": theta, "file": name,
                      "from_angle": ang[k], "to_angle": ang[k1],
                      "w_from": round(float(a1), 4), "w_to": round(float(a2), 4),
                      "PC1": round(float(pc[0]), 4), "PC2": round(float(pc[1]), 4)})
        frames.append(img)
    pd.DataFrame(mrows).to_csv(out / "morph" / "morph.csv", index=False)

    show = list(range(0, args.steps, max(1, args.steps // 12)))
    th = 192
    strip = Image.new("RGB", (len(show) * th, th + 18), "white")
    d = ImageDraw.Draw(strip)
    for j, i in enumerate(show):
        strip.paste(frames[i].resize((th, th), Image.LANCZOS), (j * th, 18))
        d.text((j * th + 4, 3), f"{mrows[i]['angle']:.0f} deg", fill="black")
    strip.save(out / "morph" / "strip.png")
    small = [f.resize((256, 256), Image.LANCZOS) for f in frames]
    small[0].save(out / "morph" / "morph.gif", save_all=True, append_images=small[1:], duration=40, loop=0)
    wr = imageio.get_writer(out / "morph" / "morph.mp4", fps=30, codec="libx264", quality=9,
                            pixelformat="yuv420p", macro_block_size=16)
    for f in frames:
        wr.append_data(np.asarray(f))
    wr.close()
    print(f"Morph: {args.steps} frames -> {out / 'morph'}")

    # ---- 3. noisy variants -------------------------------------------------
    rng = np.random.RandomState(args.noise_seed)
    if args.min_woman > 0:
        from deepface import DeepFace

        def woman_score(img):
            r = DeepFace.analyze(img_path=np.asarray(img)[:, :, ::-1].copy(), actions=("gender",),
                                 detector_backend="opencv", enforce_detection=False, silent=True)
            return float(r[0]["gender"]["Woman"])

    def noisy_woman(w0, rng):
        """One variant (1 - a) * w0 + a * w_random; redrawn until it passes --min-woman.
        Returns the saved-size image, the 256 px image, the Woman score and the number of draws."""
        draws = 0
        while True:
            draws += 1
            z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
            with torch.no_grad():
                w_rand = G.mapping(z, None, truncation_psi=1.0)
            big, sm = render(G, (1 - args.noise) * w0 + args.noise * w_rand)
            img = big.resize((args.size, args.size), Image.LANCZOS)
            score = woman_score(img) if args.min_woman > 0 else float("nan")
            if args.min_woman <= 0 or score >= args.min_woman:
                return img, sm, score, draws

    nrows = []
    for t in args.noise_around:
        i = args.angles.index(t)
        s, w0 = anchors[i], ws[i]
        folder = out / "noise" / f"{t:03.0f}deg_seed{s:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        imgs, draws = [], 0
        while len(imgs) < args.n:
            img, sm, score, k = noisy_woman(w0, rng)
            draws += k
            v = len(imgs)
            name = f"seed{s:04d}_variant_{v:03d}.png"
            img.save(folder / name)
            pc = project(sm)
            nrows.append({"anchor_angle": t, "anchor_seed": s, "variant": v, "draw": draws,
                          "file": f"{folder.name}/{name}", "woman_score": round(score, 1),
                          "PC1": round(float(pc[0]), 4), "PC2": round(float(pc[1]), 4)})
            imgs.append(sm)
        print(f"  seed {s}: kept {args.n} of {draws} draws ({draws - args.n} rejected as not a woman)")
        cols, tt = 10, 128
        grid = Image.new("RGB", (cols * tt, ((args.n + cols - 1) // cols) * tt), "white")
        for j, im in enumerate(imgs):
            grid.paste(im.resize((tt, tt), Image.LANCZOS), ((j % cols) * tt, (j // cols) * tt))
        grid.save(out / "noise" / f"sheet_{folder.name}.png")
        print(f"Noise: {args.n} variants around seed {s} ({t:.0f} deg) -> {folder}")
    ndf = pd.DataFrame(nrows)
    ndf.to_csv(out / "noise" / "variants.csv", index=False)

    # ---- 3b. one noisy variant around every morph frame --------------------
    # Consecutive morph frames are 1 deg apart and nearly identical, so a drift that walks along
    # them shows the same woman changing very slowly. Instead, each angle gets its own variant
    # around the morph face at that angle: a different woman who still looks like that point.
    # Own random stream, so the pools above do not change.
    crows = []
    if args.circle_noise:
        crng = np.random.RandomState(args.noise_seed + 1)
        cdir = out / "circle_noise" / "images"
        cdir.mkdir(parents=True, exist_ok=True)
        cimgs, total = [], 0
        for r, w0 in zip(mrows, mws):
            img, sm, score, k = noisy_woman(w0, crng)
            total += k
            name = f"circle_{r['angle']:05.1f}deg.png"
            img.save(cdir / name)
            pc = project(sm)
            crows.append({"angle": r["angle"], "file": name, "draws": k, "woman_score": round(score, 1),
                          "PC1": round(float(pc[0]), 4), "PC2": round(float(pc[1]), 4)})
            cimgs.append(sm)
        pd.DataFrame(crows).to_csv(out / "circle_noise" / "circle_noise.csv", index=False)
        cols, tt = 20, 96
        grid = Image.new("RGB", (cols * tt, ((len(cimgs) + cols - 1) // cols) * tt), "white")
        for j, im in enumerate(cimgs):
            grid.paste(im.resize((tt, tt), Image.LANCZOS), ((j % cols) * tt, (j // cols) * tt))
        grid.save(out / "circle_noise" / "sheet.png")
        print(f"Circle noise: {len(crows)} variants ({total - len(crows)} rejected as not a woman) "
              f"-> {out / 'circle_noise'}")

    # ---- PCA overview ------------------------------------------------------
    mdf = pd.DataFrame(mrows)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(P[:, 0], P[:, 1], s=25, c="lightgray", label="women_generated", zorder=1)
    colours = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for k, t in enumerate(args.noise_around):
        sub = ndf[ndf.anchor_angle == t]
        ax.scatter(sub.PC1, sub.PC2, s=14, alpha=0.6, c=colours[k % 4],
                   label=f"noise around {t:.0f} deg (n={len(sub)})", zorder=2)
    if crows:
        cdf = pd.DataFrame(crows)
        ax.scatter(cdf.PC1, cdf.PC2, s=10, alpha=0.6, c=cdf.angle, cmap="twilight",
                   marker="s", label=f"noise around each morph frame (n={len(cdf)})", zorder=2)
    ax.plot(list(mdf.PC1) + [mdf.PC1[0]], list(mdf.PC2) + [mdf.PC2[0]], "-", c="black", lw=1,
            label="morph path (re-embedded)", zorder=3)
    for t, s in zip(args.angles, anchors):
        ax.scatter(*pos[s], s=140, marker="*", c="gold", edgecolor="black", zorder=4)
        ax.annotate(f"{t:.0f}°  seed {s}", pos[s], xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, c="gray", lw=0.5)
    ax.axvline(0, c="gray", lw=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Generated-women PCA (FaceNet VGGFace2): anchors, morph, noise")
    fig.tight_layout()
    fig.savefig(out / "pca_overview.png", dpi=150)

    for t in args.noise_around:
        sub = ndf[ndf.anchor_angle == t]
        s = anchors[args.angles.index(t)]
        dist = np.hypot(sub.PC1 - pos[s][0], sub.PC2 - pos[s][1])
        print(f"  noise around {t:.0f} deg: mean PC position ({sub.PC1.mean():+.3f}, {sub.PC2.mean():+.3f}), "
              f"mean distance from anchor {dist.mean():.3f}")
    export_experiment_set(out)
    print(f"Saved everything to {out}")


if __name__ == "__main__":
    main()
