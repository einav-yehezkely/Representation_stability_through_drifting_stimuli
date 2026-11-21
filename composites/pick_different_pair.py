#################################################################
# pick_different_pair.py
#
# Find the two generated women whose facial identity differs the most,
# with pose and colour held equal, so the difference is in the features.
#
# 1. Render each candidate seed with the same frontal pose (W layers 0-2 =
#    average face) and the same colour (layers 8-17 = average face), so
#    hair / skin / eye colour cannot drive the difference.
# 2. Embed each image with the project's ResNet50-VGGFace2 model
#    (extract_embeddings/, same preprocessing as the CelebA embeddings).
# 3. Rank all pairs by cosine similarity; the lowest = most different.
#
#   py composites/pick_different_pair.py --seeds 0,5,7,...
#################################################################
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from generate_faces import load_generator, parse_seeds, seed_to_w, synth

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Adult women without glasses among seeds 0-95 (chosen by eye from candidates_sheet.png).
DEFAULT_SEEDS = "0,5,7,9,10,18,21,22,27,28,32,34,35,38,46,47,52,59,60,61,62,77,82,83,84,86,94,95"
MEAN_BGR = np.array([91.4953, 103.8827, 131.0912], np.float32)


def load_resnet(device):
    sys.path.insert(0, str(ROOT / "extract_embeddings"))
    import resnet as ResNet

    model = ResNet.resnet50(num_classes=8631, include_top=False)
    with open(ROOT / "extract_embeddings" / "resnet50_scratch_weight.pkl", "rb") as f:
        raw = pickle.loads(f.read(), encoding="latin1")
    model.load_state_dict({k: torch.from_numpy(v) for k, v in raw.items()})
    return model.to(device).eval()


@torch.no_grad()
def embed(model, img, device):
    """Resize to 256, centre-crop 224, RGB->BGR, subtract the VGGFace2 mean."""
    x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)[16:240, 16:240]
    x = x[:, :, ::-1].astype(np.float32) - MEAN_BGR
    t = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].to(device)
    v = model(t).flatten().cpu().numpy()
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--psi", type=float, default=0.7)
    ap.add_argument("--pose-layers", type=int, default=3)
    ap.add_argument("--color-from", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "generated" / "very_different"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(device)
    net = load_resnet(device)
    w_avg = G.mapping.w_avg[None, None, :]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    imgs, embs = [], []
    for s in seeds:
        w = seed_to_w(G, s, args.psi, device)
        w[:, :args.pose_layers] = w_avg
        w[:, args.color_from:] = w_avg[:, :1].expand(-1, w.shape[1] - args.color_from, -1)
        img = synth(G, w)
        imgs.append(cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA))
        embs.append(embed(net, img, device))
    E = np.stack(embs)
    sim = E @ E.T
    np.savetxt(out / "similarity.csv", sim, delimiter=",", fmt="%.4f",
               header=",".join(map(str, seeds)))

    iu = np.triu_indices(len(seeds), 1)
    order = np.argsort(sim[iu])
    print("Most different pairs (cosine similarity):")
    for k in order[:10]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {seeds[i]:3d} - {seeds[j]:3d}: {sim[i, j]:.3f}")
    print(f"(mean over all pairs {sim[iu].mean():.3f})")

    # Candidates as rendered (same pose and colour), and the top 5 pairs.
    cols, th = 7, 160
    sheet = Image.new("RGB", (cols * th, ((len(seeds) + cols - 1) // cols) * th), "white")
    d = ImageDraw.Draw(sheet)
    for n, (s, im) in enumerate(zip(seeds, imgs)):
        x, y = (n % cols) * th, (n // cols) * th
        sheet.paste(Image.fromarray(im).resize((th, th), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + 30, y + 15], fill="black")
        d.text((x + 3, y + 2), str(s), fill="white")
    sheet.save(out / "candidates_normalised.png")

    pairs = Image.new("RGB", (2 * 256 + 20, 5 * 276), "white")
    d = ImageDraw.Draw(pairs)
    for r, k in enumerate(order[:5]):
        i, j = iu[0][k], iu[1][k]
        pairs.paste(Image.fromarray(imgs[i]), (0, r * 276 + 20))
        pairs.paste(Image.fromarray(imgs[j]), (276, r * 276 + 20))
        d.text((4, r * 276 + 4), f"{seeds[i]} vs {seeds[j]}   cos = {sim[i, j]:.3f}", fill="black")
    pairs.save(out / "top_pairs.png")


if __name__ == "__main__":
    main()
