# Composite face stimuli: A/B faces and morphs

This folder builds pairs of face stimuli (category A and category B) and smooth morphs between them, in the style of averaged "composite" faces such as the Big Five trait composites. It records every approach tried so far, in order, with what worked, what did not, and why. As of 2026-11-11 the current approach is section 8.

All commands run from the repository root with the Windows Python launcher `py` (Python 3.14). Generated images, model weights and caches are git-ignored (see the end of this file).

---

## 0. Inputs and shared tools

| What | Where |
|---|---|
| Face photos (CelebA `img_align_celeba`, women, 178×218, eye-aligned) | `female_faces/` |
| 2D PCA coordinates of the ResNet50-VGGFace2 embeddings (the space used in `network_classification/`) | `pca_top2_filtered_female_vgg_1.csv` (name, PC1, PC2; 11,817 faces) |
| Full embeddings (2048-D, about 4 GB) | `female_resnet50_vggface2_embeddings.csv` |
| MediaPipe FaceLandmarker model (478 landmarks + blendshapes) | `composites/models/face_landmarker.task` |
| StyleGAN2-ADA FFHQ generator, 1024×1024 (NVIDIA; research and non-commercial licence) | `composites/models/ffhq.pkl` and code in `composites/third_party/stylegan2-ada-pytorch/` |

The scripts share these building blocks:

- **Rotation path.** As in `network_classification/tools/generate_rotation_sequence.py`, the target point is `c(θ) = r·(cos θ, sin θ)` with `r = 0.45`. A is at θ = 0° and B is at θ = 180°.
- **Identity deduplication** (`dedup_identities`). This step goes through the photos once, in order. It drops any photo whose L2-normalised embedding has cosine > 0.75 with a photo already kept. That way no woman appears many times in one average. It keeps 7,394 of the 11,817 photos.
- **Landmark morph** (`morph_two.py`). The script detects 478 MediaPipe landmarks on each image, plus 8 points on the image border. It triangulates the mean shape (Delaunay). Each frame warps both images to the interpolated shape (piecewise affine) and cross-dissolves them. Shape and texture move together.

---

## 1. Neighbourhood composites along the rotation (`make_rotation_composites.py`)

**Idea.** For each angle θ, the frame is the pixel average of the faces whose PCA position is closest to `c(θ)`. Sweeping θ gives a rotation-morph, and θ = 0° and 180° give the two opposite composites.

**Final method.**
- **Faces used.** Each frame uses the k = 120 nearest faces after deduplication.
- **Weighting.** An adaptive biweight kernel is used: R = distance to the 120th face, and `w = (1 − (d/R)²)²`. The effective number of faces per frame is about 65 (range 48–77). Across the 360 frames, 3,384 different photos are used.
- **Mask.** One fixed face-contour mask is applied (the MediaPipe FACE_OVAL of the mean frame, shrink 0.97, feather 3 px). It hides most of the hair.
- **Low-frequency equalisation** (`--equalize 5`). In every frame, the slow light and colour pattern (Gaussian σ = 5 px, computed inside the mask) is replaced by that of the mean of all frames. This removes the dark or blonde "halo" that hair colour leaves at the face edge.

**Problems found and fixed.**
| Problem | Cause | Fix |
|---|---|---|
| Flicker between frames (step difference up to 6 grey levels) | Hard top-k cut-off and per-angle deduplication | Biweight kernel and one global deduplication |
| 4 to 826 faces per frame depending on angle | Fixed-radius kernel | Adaptive radius (k-th neighbour) |
| Union-find deduplication merged too much (kept 5,412) | Transitive chains | Greedy pairwise deduplication (keeps 7,394) |
| Hair colour visible | No mask | Contour mask; an oval was tried first and rejected as too tight |
| Dark versus light rim at the face edge (spread 29.9 grey levels across angles) | Hair blurred into the edge | Low-frequency equalisation; the spread is now 4.3 |
| Per-photo masks (`--face-masks`) made faces look rounder | Averaging only the inner face of each photo | Rejected; still available as an opt-in flag |

**Result.** The final version has a mean step difference of 0.80 grey levels (maximum 2.8). Its weakness is that the A and B composites at 0° and 180° look too similar, and a glasses shadow appears around 60°–100°.

```
py composites/make_rotation_composites.py --range 360 --steps 360 --k 120 --gif
py composites/plot_rotation_circle.py --every 30        # PCA circle with example faces
py composites/make_rotation_video.py                    # large face + moving dot on the circle
```

**Output.** Everything goes to `composites/output/`: `frames/`, `frames.csv` (angle, number of faces, the faces and their weights), `A_0.png`, `B_180.png`, `rotation.gif`, `rotation_circle.png` and `rotation.mp4`.

## 2. Morph from images only (`morph_two.py`, `rotate_morph.py`)

- **`morph_two.py A B`.** This is a classic landmark morph between any two images, with no face database involved.
- **`rotate_morph.py`.** This uses N anchor images at known angles, for example 0/90/180/270. Between neighbouring anchors it interpolates spherically around their mean face (`w1 = sin(D−φ)/sin D`, `w2 = sin φ/sin D`). A 2D vector rotates the same way, so the distance from the mean stays constant. `--mode linear` gives the plain piecewise morph.

The user preferred going back to averaging at every angle (section 1). These scripts stayed as tools; `morph_two.py` is used again in section 8.

## 3. Morph between two composites only (`morph_endpoints.py`)

**Method.**
- Build only the 0° and 180° composites, unmasked.
- Process them: an unsharp mask, then shared low-frequency equalisation.
- Landmark-morph between them under one fixed contour mask.

**Result.** The transition is clean, with no intermediate identities and no glasses. However, the in-between frames are A/B blends, a straight line through the centre of the space rather than a rotation. The user also found the A/B difference hard to see.

```
py composites/morph_endpoints.py --steps 90 --gif        # -> composites/endpoint_morph/
```

## 4. A and B with a third face as noise (`make_noise_stimuli.py`)

**Method.**
- The 90° composite N serves as the noise face.
- For each side X in {A, B}, 50 images are made as `(1−a)·X + a·N` (landmark morph), with `a` evenly spaced from 0 to 0.98. A and B converge to the same face as `a` grows.
- Output: 100 images, `stimuli.csv`, and `overview.png`.

The base A/B difference was still small, and the 90° composite carries the glasses shadow.

```
py composites/make_noise_stimuli.py --n 50              # -> composites/noise_stimuli/
```

## 5. Shape-normalised composites in the Big Five style (`make_bigfive_style.py`)

This section reproduces how the classic trait composites are made.

**Method.**
1. Screen the 500 faces nearest each angle. Keep only neutral, frontal faces: MediaPipe blendshapes smile ≤ 0.25 and jawOpen ≤ 0.15, and nose offset from the eye midpoint ≤ 0.12 of the eye distance. Take the nearest 100 that pass.
2. Warp each face to the group's mean landmark shape, then average. The eyes, nose and mouth line up, so they stay sharp, and hair and background blur into a soft halo.
3. No mask; save a square crop at 512×512.

**Result.** These are the closest to the reference look, and A and B are clearly different women (blonde with light eyes versus dark-haired with dark eyes). However, hair colour is fully visible and is a large part of the difference.

```
py composites/make_bigfive_style.py --angles 0 180 --k 100   # -> composites/bigfive_style/
```

## 6. Synthetic faces from StyleGAN2 (`generate_faces.py`)

The user wanted two clearly different women made independently of the database, so faces were generated with StyleGAN2-ADA (FFHQ), truncation ψ = 0.7.

- The CUDA plug-ins cannot be compiled on this machine (no MSVC). The script therefore forces the reference ops, which give the same output but run slower.
- A sheet of 96 candidates (seeds 0–95) was generated. The user picked seeds **0** and **83**.

```
py composites/generate_faces.py --seeds 0-95       # -> composites/generated/candidates_sheet.png
py composites/generate_faces.py --pick 0 83        # full-size 1024 px + W latents
```

## 7. Many noisy variants around each face, averaged (`noise_average_generated.py`)

**Idea.** Make a composite around one synthetic woman: generate many similar women around her, then average them.

**Noise in the latent W space.**
- Each variant is `w_i = (1 − a)·w_seed + a·w_random_i`, where `w_random_i` is the latent of a random face (ψ = 1).
- Every variant is therefore a realistic face that resembles the original.
- 100 variants are made per side. They are saved as `seed<NNNN>/seed<NNNN>_variant_<iii>.png`, so the two sides never share a file name. Runs made before 2026-11-17 used plain `variant_<iii>.png`; in `pair_38_46_hairvar/` those files have been renamed to the new format.

**Averaging.** As in section 5, each variant is warped to the side's mean landmark shape and the variants are averaged. A plain pixel average is saved as well.

| Run (folder under `composites/generated/noise_average/`) | Settings | Mean A–B pixel difference | Outcome |
|---|---|---|---|
| `noise_0.50/` | a = 0.5 | 19.3 | Both averages drift toward the generic mean face, so they are too similar. Some variants are men or children. |
| `noise_0.30/` | a = 0.3 | 30.9 | Clearer difference and all variants are women, but the morph mostly showed the head turning (seed 0 faces right, seed 83 faces left). |
| `noise_0.30_pose3/` | a = 0.3, pose fixed | 23.9 | All faces frontal; the morph changes only features. The user found the feature differences too small. |
| `noise_0.30_pose3_ex1.5/` | as above, exaggeration ×1.5 | 35.0 | Bigger difference. |
| `noise_0.30_pose3_ex2/` | as above, exaggeration ×2 | 45.4 | Biggest difference, but mostly hair colour (red-brown vs. dark). |
| `noise_0.30_pose3_ex2_col8/` | as above, colour layers 8–17 shared | 23.5 | **Same hair and skin colour** (hair RGB 86/63/56 vs. 80/60/53); the remaining differences are face shape, eyes, brows, nose and hair length. |

**Pose fix (`--pose-layers 3`, now the default).**
- Head pose lives in StyleGAN's coarse W layers. Layers 0–2 of every latent (the originals and all variants) are replaced by the generator's average latent `w_avg`, which is frontal and upright.
- Head turn (nose offset from the eye midpoint relative to the eye distance; 0 = straight):

  | | before | after |
  |---|---|---|
  | seed 0 | +0.086 | +0.022 |
  | seed 83 | −0.038 | 0.000 |

- Testing 2–5 layers showed that 3 is enough. With 4 or more layers, hair and apparent age change too.
- Cost: those layers also carry some face shape, so the two faces become more alike in proportions.

**Exaggeration (`--exaggerate s`).**
- Before noise is added, the two original latents are pushed apart around their mean: `w = mean + s·(w − mean)`, a caricature in latent space.
- The pose layers are identical for both faces, so this changes only features, skin, eyes, brows and hair.

**Same hair colour (`--color-from 8`).**
- The fine W layers (8–17 of 18) carry colour: hair, skin and eye colour.
- Every latent, the two originals and all variants, gets the same layers 8–17, namely the mean of the two seeds'.
- Testing start layers 10, 8, 7, 6 and 5 showed that 10 still leaves the hair colours different. At 8 the colours match while the features stay distinct. At 6 or lower the features begin to merge, for example both pairs of eyes become alike.

**Morphs.** Each run folder has a `morph/` subfolder made with `morph_two.py --no-mask --steps 90`. It holds 91 frames, `morph.gif` (back and forth), `morph.mp4` (512×512, 30 fps), `strip.png` and `landmarks_check.png`.

```
py composites/noise_average_generated.py --seeds 0 83 --n 100 --noise 0.3 --pose-layers 3 --exaggerate 2 --color-from 8
py composites/morph_two.py <run>/average_seed0000.png <run>/average_seed0083.png --no-mask --steps 90 --gif --out <run>/morph
```

## 7b. A pair chosen to be very different (`pick_different_pair.py`)

Seeds 0 and 83 stayed too similar, so the pair was chosen by measurement instead of by eye.

**Method.**
1. Render the 28 adult women without glasses among seeds 0–95 with the same frontal pose (layers 0–2) and the same colour (layers 8–17), both taken from the average face.
2. Embed each image with the project's ResNet50-VGGFace2 model, using the same preprocessing as the CelebA embeddings.
3. Rank all 378 pairs by cosine similarity. The mean over all pairs is 0.525.

**Result.**
- The most different pairs: 7–38 (0.197), 38–47 (0.241), 38–46 (0.247) and 38–82 (0.248).
- **38–46** was chosen. Its similarity is almost as low as the top pair, and both women have a similar, fairly neutral expression. Pair 7–38 also differs strongly in age and smile.
- The pair was then averaged as in section 7: 100 variants per side, a = 0.3, pose layers 3, colour layers from 8, no exaggeration. Then it was morphed.
- The mean A–B pixel difference is 25.8.
- The differences now include face shape, eyes, brows, nose, apparent ethnicity and age, and the fringe (bangs) of seed 38.

```
py composites/pick_different_pair.py         # -> generated/very_different/top_pairs.png, similarity.csv
py composites/noise_average_generated.py --seeds 38 46 --n 100 --noise 0.3 --pose-layers 3 --color-from 8 --out composites/generated/very_different/pair_38_46
```

**Version of 38 without the fringe (`--edit 38:83:3,4`).** The hairstyle (fringe) lives in W layer 4, and partly in layer 3.

- **Tests.** Layers 3, 4, 5, 3–4, 4–5 and 3–5 were copied from four donors: the average face and seeds 46, 83 and 0.
  - Layers 3 or 5 alone leave the fringe.
  - Layer 4 alone removes it only partly.
  - Layers 3–4 from seed 83 remove it fully (centre parting, forehead visible) and keep the most identity: cosine 0.70 to the original 38. Other donors give 0.47–0.67.
- **Results.** Output is in `generated/very_different/pair_38nofringe_46/`.
  - The averages of 38 and of 38-without-fringe have cosine 0.76.
  - The no-fringe 38 is still very different from 46: cosine 0.44, compared with 0.37 with the fringe.
- **Side effect.** Seed 83's layers bring a smile into 38.

**100 variants per woman with a varied hairstyle (`pair_38_46_hairvar/`).** The user preferred `pair_38_46` (38 with the fringe), but did not want every variant around 38 to have a fringe.

- **More hairstyle noise:** `--hair-layers 4 --hair-noise 0.8`. Layer 4, the hairstyle layer, gets 80% noise while the other layers keep 30%.
- **Side effect:** layer 4 also carries gender and age, so with fully random noise faces many variants became men or children.
- **Fix:** `--noise-pool <26 adult women without glasses> --pool-mix 3`. The noise face is now a random convex mix (Dirichlet weights) of 3 women from the pool of section 7b, excluding 38 and 46. After this, all 200 variants are adult women.
- **Same procedure on both sides.** 46 gets exactly the same treatment.
- **Trade-off:** the pool women mostly smile, so the variants smile more than the originals. Mixing 3 pool faces also lowers the spread between variants.

```
py composites/noise_average_generated.py --seeds 38 46 --n 100 --noise 0.3 --pose-layers 3 --color-from 8 --hair-layers 4 --hair-noise 0.8 --noise-pool "0,5,7,9,10,18,21,22,27,28,32,34,35,38,46,47,52,59,60,61,62,77,82,83,84,86,94,95" --out composites/generated/very_different/pair_38_46_hairvar
```

## 7c. A closed circle through 38 and 46 (`latent_circle.py`)

This uses the same geometry as the PCA rotation, but in StyleGAN's W space: `w(θ) = c + cos θ · a + sin θ · b`.

- **The terms.**
  - `c` is the midpoint of 38 and 46.
  - `a = w_38 − c`, so 38 sits at 0° and 46 at 180°.
  - `b` is the direction from `c` to a third woman, made orthogonal to `a` and scaled to `|a|`.
- **The third woman.** By default it is the pool woman most orthogonal to the 38–46 axis; for 38/46 this is seed 61.
- **The two routes.** 0° → 180° runs from 38 to 46 through `c + b`. 180° → 360° runs back from 46 to 38 through `c − b`, so the return frames are different faces.
- **Fixed settings.** Pose (layers 0–2) and colour (layers 8–17) are fixed as in section 7.

**Output** (`pair_38_46_hairvar/circle/`):
- `38_to_46/`: 180 frames, θ = 0–179°.
- `46_to_38/`: 180 frames, θ = 180–359°.
- Each frame is named `circle_<θ>deg.png`.
- `circle.csv` gives each frame's angle and its identity similarity (ResNet50-VGGFace2) to 38 and to 46.
- `strip.png`, `circle.gif` and `circle.mp4`.

**Identity similarity around the circle.**

| θ | 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 |
|---|---|---|---|---|---|---|---|---|
| to 38 | 1.00 | 0.86 | 0.42 | 0.25 | 0.26 | 0.35 | 0.71 | 0.93 |
| to 46 | 0.26 | 0.27 | 0.45 | 0.81 | 1.00 | 0.81 | 0.47 | 0.29 |

The circle is not symmetric in identity: the 270° side stays closer to 38 than the 90° side does.

```
py composites/latent_circle.py --seeds 38 46 --out composites/generated/very_different/pair_38_46_hairvar/circle
```

## 8. Current state and open points

- **Current best candidate:** `composites/generated/noise_average/noise_0.30_pose3_ex2_col8/`.
  - Both faces are frontal and have the same hair and skin colour.
  - They differ in facial features: a rounder versus a narrower face, and different eyes, brows and nose. Hair length differs too.
  - The morph is smooth.
- **Face-shape difference is limited,** because the pose fix (layers 0–2) also fixes some of the face shape. To get more:
  - choose seed pairs that differ more in shape;
  - or exaggerate the feature layers (3–7) more strongly.
- **Hair length and style still differ** (hair colour is now the same). A contour mask, as in section 1, would hide this.
- **Both originals smile,** so the averages smile. The reference composites are neutral. Options are an expression direction in W, or filtering variants by the MediaPipe smile score as in section 5.
- **The generated faces have not been placed in the project's PCA space** (deferred on request). A PCA recomputed from the full embeddings CSV (`composites/cache/pca_resnet50.npz`) does **not** reproduce `pca_top2_filtered_female_vgg_1.csv`: the correlation is only 0.87 / 0.81. The exact preprocessing behind that file needs checking first.
- **Nothing is committed yet.**

## Files

| Script | Purpose | Default output |
|---|---|---|
| `make_rotation_composites.py` | Neighbourhood composites along the PCA rotation | `output/` |
| `plot_rotation_circle.py` | PCA circle with composite thumbnails | `output/rotation_circle.png` |
| `make_rotation_video.py` | Video: large face and position on the circle | `output/rotation.mp4` |
| `morph_two.py` | Landmark morph between two images | `morph_output/` |
| `rotate_morph.py` | Circular morph through N anchor images | `rotation_output/` |
| `morph_endpoints.py` | Morph between the 0° and 180° composites only (shared helpers) | `endpoint_morph/` |
| `make_noise_stimuli.py` | A and B with the 90° composite as noise | `noise_stimuli/` |
| `make_bigfive_style.py` | Shape-normalised neutral composites | `bigfive_style/` |
| `generate_faces.py` | StyleGAN2 candidate sheet and full-size picks | `generated/` |
| `noise_average_generated.py` | Noisy latent variants, averaged | `generated/noise_average/<run>/` |
| `latent_circle.py` | Closed circle of faces through two women in W space | `<out>/<A>_to_<B>/`, `<out>/<B>_to_<A>/` |
| `pick_different_pair.py` | Rank generated women by identity difference (ResNet50-VGGFace2) | `generated/very_different/` |

**Git-ignored folders** (they can be regenerated): `cache/`, `models/`, `third_party/`, `output/`, `endpoints/`, `endpoint_morph/`, `morph_output/`, `anchors/`, `rotation_output*/`, `noise_stimuli/`, `bigfive_style/`, `generated/`.

To set up on a new machine:

```
git clone --depth 1 https://github.com/NVlabs/stylegan2-ada-pytorch.git composites/third_party/stylegan2-ada-pytorch
curl -L -o composites/models/ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

The MediaPipe model `face_landmarker.task` also goes in `composites/models/`.
