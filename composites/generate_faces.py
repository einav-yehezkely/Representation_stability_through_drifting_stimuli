#################################################################
# generate_faces.py
#
# Synthetic (non-existing) faces from the pretrained StyleGAN2-ADA FFHQ
# generator (NVIDIA, research/non-commercial licence).
#
#   py composites/generate_faces.py --seeds 0-95          # candidate sheet
#   py composites/generate_faces.py --pick 12 57          # full-size images of two
#
# Needs composites/third_party/stylegan2-ada-pytorch (code) and
# composites/models/ffhq.pkl (weights).
#################################################################
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party" / "stylegan2-ada-pytorch"))
warnings.filterwarnings("ignore")


def load_generator(device):
    from torch_utils.ops import bias_act, upfirdn2d
    bias_act._init = lambda: False      # no CUDA compiler here: use the
    upfirdn2d._init = lambda: False     # reference ops (same output, slower)
    with open(HERE / "models" / "ffhq.pkl", "rb") as f:
        return pickle.load(f)["G_ema"].to(device).eval()


@torch.no_grad()
def seed_to_w(G, seed, psi, device):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    return G.mapping(z, None, truncation_psi=psi)


@torch.no_grad()
def synth(G, w):
    img = G.synthesis(w, noise_mode="const")
    img = (img.clamp(-1, 1) + 1) * 127.5
    return img[0].permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-95", help="candidate seeds, e.g. 0-95 or 3,7,10-20")
    ap.add_argument("--pick", type=int, nargs="+", help="seeds to render at full size")
    ap.add_argument("--psi", type=float, default=0.7, help="truncation (lower = more typical)")
    ap.add_argument("--out", default=str(HERE / "generated"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.pick:
        for s in args.pick:
            w = seed_to_w(G, s, args.psi, device)
            Image.fromarray(synth(G, w)).save(out / f"seed{s:04d}.png")
            np.save(out / f"seed{s:04d}_w.npy", w.cpu().numpy())
        print(f"Saved {len(args.pick)} images to {out}")
        return

    seeds = parse_seeds(args.seeds)
    thumb, cols = 160, 12
    rows = (len(seeds) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    (out / "candidates").mkdir(exist_ok=True)
    for i, s in enumerate(seeds):
        img = Image.fromarray(synth(G, seed_to_w(G, s, args.psi, device)))
        img.resize((256, 256), Image.LANCZOS).save(out / "candidates" / f"seed{s:04d}.png")
        x, y = (i % cols) * thumb, (i // cols) * thumb
        sheet.paste(img.resize((thumb, thumb), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + 34, y + 16], fill="black")
        draw.text((x + 3, y + 2), str(s), fill="white")
    sheet.save(out / "candidates_sheet.png")
    print(f"Saved {len(seeds)} candidates to {out / 'candidates'} and candidates_sheet.png")


if __name__ == "__main__":
    main()
