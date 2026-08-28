import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

ARCFACE_CSV = "female_arcface_embeddings.csv"

# This must be the FaceNet PCA file used to define the rotation
PCA_CSV = "pca_top2_filtered_female.csv"

FILENAME_COL = "filename"
PC1_COL = "PC1"
PC2_COL = "PC2"

WINDOW_DEG = 20.0
STEP_DEG = 1.0

OUTPUT_CSV = "arcface_centroid_smoothness_20deg.csv"
OUTPUT_PLOT = "arcface_centroid_smoothness_20deg.png"


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("=" * 70)
print("LOADING ARCFACE EMBEDDINGS")
print("=" * 70)

arc_df = pd.read_csv(
    ARCFACE_CSV,
    header=None
)

print("ArcFace shape:", arc_df.shape)

if arc_df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns "
        f"(filename + 512 embedding values), "
        f"got {arc_df.shape[1]}"
    )


embedding_lookup = {}

for _, row in arc_df.iterrows():

    filename = os.path.basename(
        str(row.iloc[0]).strip()
    )

    emb = row.iloc[1:513].to_numpy(
        dtype=np.float32
    )

    norm = np.linalg.norm(emb)

    if norm > 0:
        emb = emb / norm

    embedding_lookup[filename] = emb


print(
    "Loaded ArcFace embeddings:",
    len(embedding_lookup)
)


# ============================================================
# LOAD FACENET PCA
# ============================================================

print()
print("=" * 70)
print("LOADING FACENET PCA")
print("=" * 70)

pca_df = pd.read_csv(
    PCA_CSV,
    header=None,
    names=[
        "filename",
        "PC1",
        "PC2"
    ]
)

print(
    "Columns:",
    list(pca_df.columns)
)

print(
    "Rows:",
    len(pca_df)
)


# ============================================================
# HANDLE POSSIBLE COLUMN NAMES
# ============================================================

if FILENAME_COL not in pca_df.columns:

    possible_filename_cols = [
        c for c in pca_df.columns
        if "file" in c.lower()
        or "image" in c.lower()
        or "name" in c.lower()
    ]

    if not possible_filename_cols:
        raise RuntimeError(
            "Could not find filename column."
        )

    FILENAME_COL = possible_filename_cols[0]


if PC1_COL not in pca_df.columns:

    possible_pc1 = [
        c for c in pca_df.columns
        if c.lower() in [
            "pc1",
            "pca1",
            "pca_1"
        ]
    ]

    if not possible_pc1:
        raise RuntimeError(
            "Could not find PC1 column."
        )

    PC1_COL = possible_pc1[0]


if PC2_COL not in pca_df.columns:

    possible_pc2 = [
        c for c in pca_df.columns
        if c.lower() in [
            "pc2",
            "pca2",
            "pca_2"
        ]
    ]

    if not possible_pc2:
        raise RuntimeError(
            "Could not find PC2 column."
        )

    PC2_COL = possible_pc2[0]


print(
    "Using filename column:",
    FILENAME_COL
)

print(
    "Using PC1 column:",
    PC1_COL
)

print(
    "Using PC2 column:",
    PC2_COL
)


# ============================================================
# CALCULATE ANGLE FROM FACENET PCA
# ============================================================

pca_df = pca_df.copy()

pca_df["filename_clean"] = (
    pca_df[FILENAME_COL]
    .astype(str)
    .apply(os.path.basename)
)


# atan2 gives angle in [-180, 180]
angles = np.degrees(
    np.arctan2(
        pca_df[PC2_COL].to_numpy(),
        pca_df[PC1_COL].to_numpy()
    )
)

# convert to [0, 360)
pca_df["angle_deg"] = (
    angles + 360
) % 360


# ============================================================
# KEEP ONLY IMAGES THAT HAVE ARCFACE EMBEDDINGS
# ============================================================

pca_df = pca_df[
    pca_df["filename_clean"].isin(
        embedding_lookup
    )
].copy()


print()
print(
    "Images with both FaceNet PCA "
    "and ArcFace embedding:",
    len(pca_df)
)


# ============================================================
# CIRCULAR ANGULAR DISTANCE
# ============================================================

def circular_distance(
    angles,
    center_angle
):

    diff = np.abs(
        angles - center_angle
    )

    return np.minimum(
        diff,
        360 - diff
    )


# ============================================================
# BUILD ONE ARCFACE CENTROID FOR EACH 20° WINDOW
#
# Example:
# center = 100°
# window = [90°, 110°]
#
# Then center moves by 1°.
# ============================================================

centroids = []
window_info = []


all_angles = pca_df[
    "angle_deg"
].to_numpy()


centers = np.arange(
    0,
    360,
    STEP_DEG
)


half_window = WINDOW_DEG / 2


for center in centers:

    distances = circular_distance(
        all_angles,
        center
    )

    mask = distances <= half_window

    window_df = pca_df[
        mask
    ]


    embeddings = []

    for filename in window_df[
        "filename_clean"
    ]:

        embeddings.append(
            embedding_lookup[
                filename
            ]
        )


    if len(embeddings) == 0:

        centroids.append(
            None
        )

        window_info.append({
            "center_angle":
                center,
            "num_images":
                0
        })

        continue


    embeddings = np.stack(
        embeddings
    )


    # Mean ArcFace vector
    centroid = embeddings.mean(
        axis=0
    )


    # Normalize centroid
    centroid_norm = np.linalg.norm(
        centroid
    )

    if centroid_norm > 0:
        centroid = (
            centroid /
            centroid_norm
        )


    centroids.append(
        centroid
    )


    window_info.append({
        "center_angle":
            center,
        "num_images":
            len(embeddings)
    })


# ============================================================
# COMPARE CONSECUTIVE WINDOWS
#
# 0° window vs 1° window
# 1° window vs 2° window
# ...
# 359° window vs 0° window
# ============================================================

results = []


for i in range(
    len(centers)
):

    j = (
        i + 1
    ) % len(centers)


    c1 = centroids[i]
    c2 = centroids[j]


    if (
        c1 is None
        or c2 is None
    ):

        continue


    cosine = float(
        np.dot(
            c1,
            c2
        )
    )


    euclidean = float(
        np.linalg.norm(
            c1 - c2
        )
    )


    results.append({

        "angle_current":
            centers[i],

        "angle_next":
            centers[j],

        "num_images_current":
            window_info[i][
                "num_images"
            ],

        "num_images_next":
            window_info[j][
                "num_images"
            ],

        "cosine_similarity":
            cosine,

        "euclidean_distance":
            euclidean
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("20° WINDOW CENTROID SMOOTHNESS")
print("=" * 70)

print(
    "Mean cosine similarity:",
    results_df[
        "cosine_similarity"
    ].mean()
)

print(
    "Median cosine similarity:",
    results_df[
        "cosine_similarity"
    ].median()
)

print(
    "Minimum cosine similarity:",
    results_df[
        "cosine_similarity"
    ].min()
)

print()

print(
    "Mean Euclidean distance:",
    results_df[
        "euclidean_distance"
    ].mean()
)

print(
    "Median Euclidean distance:",
    results_df[
        "euclidean_distance"
    ].median()
)

print(
    "Maximum Euclidean distance:",
    results_df[
        "euclidean_distance"
    ].max()
)

print()

print(
    "Mean images per window:",
    np.mean(
        [
            x["num_images"]
            for x in window_info
        ]
    )
)


# ============================================================
# WORST TRANSITIONS
# ============================================================

print()
print("=" * 70)
print("20 LEAST SIMILAR CONSECUTIVE WINDOWS")
print("=" * 70)

worst = results_df.sort_values(
    "cosine_similarity"
).head(20)


for _, row in worst.iterrows():

    print(
        f'{row["angle_current"]:.0f}° '
        f'-> '
        f'{row["angle_next"]:.0f}° '
        f'| cosine = '
        f'{row["cosine_similarity"]:.6f} '
        f'| distance = '
        f'{row["euclidean_distance"]:.6f} '
        f'| N = '
        f'{int(row["num_images_current"])}'
    )


# ============================================================
# RANDOM WINDOW CENTROID BASELINE
#
# Compare each centroid with a randomly selected,
# non-neighboring centroid.
# ============================================================

rng = np.random.default_rng(
    42
)

random_cosines = []


valid_indices = [
    i
    for i, c in enumerate(
        centroids
    )
    if c is not None
]


for i in valid_indices:

    possible = [
        j
        for j in valid_indices
        if circular_distance(
            np.array(
                [centers[j]]
            ),
            centers[i]
        )[0] > WINDOW_DEG
    ]


    if not possible:
        continue


    j = rng.choice(
        possible
    )


    random_cosines.append(
        float(
            np.dot(
                centroids[i],
                centroids[j]
            )
        )
    )


print()
print("=" * 70)
print("CONSECUTIVE VS RANDOM WINDOWS")
print("=" * 70)

print(
    "Mean consecutive centroid cosine:",
    results_df[
        "cosine_similarity"
    ].mean()
)

print(
    "Mean random centroid cosine:",
    np.mean(
        random_cosines
    )
)

print(
    "Difference:",
    results_df[
        "cosine_similarity"
    ].mean()
    -
    np.mean(
        random_cosines
    )
)


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print(
    "Saved:",
    OUTPUT_CSV
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    results_df[
        "angle_current"
    ],
    results_df[
        "cosine_similarity"
    ]
)

plt.axhline(
    results_df[
        "cosine_similarity"
    ].mean(),
    linestyle="--",
    label="Mean"
)

plt.xlabel(
    "FaceNet-PCA angle"
)

plt.ylabel(
    "ArcFace centroid cosine similarity"
)

plt.title(
    "ArcFace Smoothness Between Consecutive 20° "
    "FaceNet-PCA Windows"
)

plt.ylim(
    -0.1,
    1.01
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.show()