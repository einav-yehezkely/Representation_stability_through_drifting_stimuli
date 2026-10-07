"""
Generate parity fixtures: reference outputs computed with the source experiment's own
functions (verbatim copies in source_functions.py), used by tests/unit/parity.test.js to
check that the JavaScript port reproduces them.

The human-experiment selection rules (researcher decisions ה-1 .. ה-5 in IMPLEMENTATION_PLAN.md)
are expressed with the source primitives only:
  * cluster of size 1  -> collect_nearest_images(center, k=1)
  * "nearest not yet used" -> the masking used in generate_rotation_sequence
    (dists[used] = inf, then argmin)
  * centers move with rotate_vector after each drift response (as in the source main loop)
  * B = -A (create_base_and_opposite_points)

Usage (from human_experiment/):  py -3 scripts/reference/make_parity_fixtures.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(PROJECT, ".."))
sys.path.insert(0, HERE)

from source_functions import (  # noqa: E402
    collect_nearest_images,
    create_base_and_opposite_points,
    generate_rotation_sequence,
    load_top2_filtered,
    rotate_vector,
)

PCA_CSV = os.path.join(REPO, "pca_top2_filtered_female_vgg_1.csv")  # read-only
OUT = os.path.join(PROJECT, "tests", "fixtures", "parity", "parity.json")

names, points = load_top2_filtered(PCA_CSV)
name_to_index = {name: i for i, name in enumerate(names)}


def nearest_unused(center, used):
    dists = np.linalg.norm(points - center, axis=1)
    for idx in used:
        dists[idx] = np.inf
    return int(np.argmin(dists)), float(dists[np.argmin(dists)])


def base_index(base_point):
    matches = np.where((points[:, 0] == base_point[0]) & (points[:, 1] == base_point[1]))[0]
    return int(matches[0])


fixture = {"pcaRowCount": int(len(names))}

# 1. Angle convention (ANGLE_MAP, classify_rotation_resnet50.py:52-56) for a sample of points.
sample = list(range(0, len(names), 97))
angle_map = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
fixture["angles"] = {"indices": sample, "angleDeg": [float(angle_map[i]) for i in sample]}

# 2. Base and opposite points.
fixture["basePoints"] = []
for angle in [0, 37, 180, 271.5]:
    base, opposite = create_base_and_opposite_points(angle, PCA_CSV)
    fixture["basePoints"].append(
        {
            "angle": angle,
            "index": base_index(base),
            "base": [float(base[0]), float(base[1])],
            "opposite": [float(opposite[0]), float(opposite[1])],
        }
    )

base0, opp0 = create_base_and_opposite_points(0, PCA_CSV)

# 3. Incremental rotation (source main loop: base_point = rotate_vector(base_point, ROTATION_DEGS)).
fixture["rotations"] = []
for step in [1.0, 0.1, 0.5]:
    for n in [1, 90, 360]:
        v = base0.copy()
        for _ in range(n):
            v = rotate_vector(v, step)
        fixture["rotations"].append({"step": step, "n": n, "start": base0.tolist(), "end": v.tolist()})

# 4. k-nearest clusters (general k) at several centers.
fixture["clusters"] = []
for angle in [0, 45, 133, 250]:
    center = rotate_vector(base0, angle)
    for k in [1, 5, 64]:
        idx, _ = collect_nearest_images(center, points, names, k=k)
        fixture["clusters"].append({"center": center.tolist(), "k": k, "indices": [int(i) for i in idx]})

# 5. Training sequence: fixed centers, nearest image not yet used in training (one set for the phase).
rng = np.random.RandomState(0)
training_groups = ["A"] * 60 + ["B"] * 60
rng.shuffle(training_groups)
used = set()
training = []
for group in training_groups:
    center = base0 if group == "A" else opp0
    idx, dist = nearest_unused(center, used)
    used.add(idx)
    training.append({"group": group, "index": idx, "distance": dist})
fixture["training"] = {"groups": training_groups, "trials": training}

# 6. Drift sequences: centers start at the training centers, selection = nearest not yet used in
#    drift (fresh set), centers rotate after each response.
fixture["drift"] = []
for step, n_trials in [(1.0, 360), (0.5, 200)]:
    rng = np.random.RandomState(1)
    groups = ["A"] * (n_trials // 2) + ["B"] * (n_trials // 2)
    rng.shuffle(groups)
    center_a, center_b = base0.copy(), opp0.copy()
    used = set()
    trials = []
    for group in groups:
        center = center_a if group == "A" else center_b
        idx, dist = nearest_unused(center, used)
        used.add(idx)
        trials.append(
            {
                "group": group,
                "index": idx,
                "distance": dist,
                "centerA": center_a.tolist(),
                "centerB": center_b.tolist(),
            }
        )
        center_a = rotate_vector(center_a, step)
        center_b = rotate_vector(center_b, step)
    fixture["drift"].append({"step": step, "groups": groups, "trials": trials, "finalCenterA": center_a.tolist()})

# 7. Source trajectory sequence (used by the visualization as an optional layer).
seq, _ = generate_rotation_sequence(base0, points, names, num_steps=360, start_angle=0, rotation_range=360, used_indices=set())
fixture["rotationSequence"] = [{"step": int(s), "angleDeg": float(a), "index": name_to_index[n]} for s, a, n in seq]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(fixture, f)
print(f"Wrote {os.path.relpath(OUT, PROJECT)}")
