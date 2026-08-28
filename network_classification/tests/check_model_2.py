import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

EMBEDDINGS_CSV = "female_arcface_embeddings.csv"

TRAJECTORY_CSV = "rotation_sequence_all.csv"

# change these if your column names are different
FILENAME_COL = "filename"
ANGLE_COL = "angle_deg"

OUTPUT_CSV = "arcface_trajectory_smoothness.csv"
OUTPUT_PLOT = "arcface_trajectory_smoothness.png"


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("=" * 70)
print("LOADING ARCFACE EMBEDDINGS")
print("=" * 70)

emb_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None
)

if emb_df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns "
        f"(filename + 512 embedding values), "
        f"got {emb_df.shape[1]}"
    )


embedding_lookup = {}

for _, row in emb_df.iterrows():

    filename = str(
        row.iloc[0]
    ).strip()

    basename = os.path.basename(
        filename
    )

    emb = row.iloc[
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    # L2 normalization
    norm = np.linalg.norm(
        emb
    )

    if norm > 0:
        emb = emb / norm

    embedding_lookup[
        basename
    ] = emb


print(
    "Loaded embeddings:",
    len(embedding_lookup)
)


# ============================================================
# LOAD TRAJECTORY
# ============================================================

print()
print("=" * 70)
print("LOADING TRAJECTORY")
print("=" * 70)

traj = pd.read_csv(
    TRAJECTORY_CSV
)

print(
    "Columns:",
    list(traj.columns)
)

print(
    "Rows:",
    len(traj)
)


if FILENAME_COL not in traj.columns:
    raise RuntimeError(
        f"Could not find column "
        f"'{FILENAME_COL}'"
    )

if ANGLE_COL not in traj.columns:
    raise RuntimeError(
        f"Could not find column "
        f"'{ANGLE_COL}'"
    )


# ============================================================
# ATTACH ARCFACE EMBEDDINGS
# ============================================================

valid_rows = []
missing = []


for _, row in traj.iterrows():

    filename = os.path.basename(
        str(
            row[FILENAME_COL]
        ).strip()
    )

    if filename not in embedding_lookup:

        missing.append(
            filename
        )

        continue


    valid_rows.append({
        "filename": filename,
        "angle": float(
            row[ANGLE_COL]
        ),
        "embedding":
            embedding_lookup[
                filename
            ]
    })


print(
    "Valid trajectory images:",
    len(valid_rows)
)

print(
    "Missing embeddings:",
    len(missing)
)

if missing:
    print(
        "Examples:",
        missing[:10]
    )


# ============================================================
# SORT BY ANGLE
# ============================================================

valid_rows = sorted(
    valid_rows,
    key=lambda x: x["angle"]
)


# ============================================================
# CASE 1:
# ONE IMAGE PER TRAJECTORY STEP
#
# Compare each embedding with the next embedding.
# ============================================================

results = []


for i in range(
    len(valid_rows) - 1
):

    current = valid_rows[i]
    nxt = valid_rows[i + 1]


    emb1 = current[
        "embedding"
    ]

    emb2 = nxt[
        "embedding"
    ]


    # Since embeddings are L2 normalized,
    # dot product = cosine similarity
    cos_sim = float(
        np.dot(
            emb1,
            emb2
        )
    )


    euclidean = float(
        np.linalg.norm(
            emb1 - emb2
        )
    )


    results.append({

        "filename_current":
            current["filename"],

        "filename_next":
            nxt["filename"],

        "angle_current":
            current["angle"],

        "angle_next":
            nxt["angle"],

        "angle_difference":
            nxt["angle"]
            - current["angle"],

        "cosine_similarity":
            cos_sim,

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
print("CONSECUTIVE ARCFACE SIMILARITY")
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
    "Min cosine similarity:",
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
    "Max Euclidean distance:",
    results_df[
        "euclidean_distance"
    ].max()
)


# ============================================================
# PRINT WORST JUMPS
# ============================================================

print()
print("=" * 70)
print("20 LARGEST JUMPS")
print("=" * 70)

worst = results_df.sort_values(
    "cosine_similarity",
    ascending=True
).head(20)


for _, row in worst.iterrows():

    print(
        f'{row["angle_current"]:.2f}° '
        f'-> '
        f'{row["angle_next"]:.2f}° '
        f'| '
        f'{row["filename_current"]} '
        f'-> '
        f'{row["filename_next"]} '
        f'| cosine = '
        f'{row["cosine_similarity"]:.4f} '
        f'| euclidean = '
        f'{row["euclidean_distance"]:.4f}'
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
    figsize=(12, 5)
)

plt.plot(
    results_df[
        "angle_current"
    ],
    results_df[
        "cosine_similarity"
    ]
)

plt.xlabel(
    "Rotation angle"
)

plt.ylabel(
    "Cosine similarity to next step"
)

plt.title(
    "ArcFace Smoothness Along FaceNet-PCA Rotation"
)

plt.axhline(
    results_df[
        "cosine_similarity"
    ].mean(),
    linestyle="--",
    label="Mean"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.show()


# ============================================================
# OPTIONAL:
# COMPARE WITH RANDOM IMAGE PAIRS
# ============================================================

embeddings_matrix = np.stack(
    [
        x["embedding"]
        for x in valid_rows
    ]
)

rng = np.random.default_rng(
    42
)

num_pairs = min(
    5000,
    len(valid_rows) * 5
)

random_cosines = []


for _ in range(
    num_pairs
):

    i, j = rng.choice(
        len(valid_rows),
        size=2,
        replace=False
    )

    random_cosines.append(
        float(
            np.dot(
                embeddings_matrix[i],
                embeddings_matrix[j]
            )
        )
    )


random_mean = np.mean(
    random_cosines
)


trajectory_mean = results_df[
    "cosine_similarity"
].mean()


print()
print("=" * 70)
print("TRAJECTORY VS RANDOM")
print("=" * 70)

print(
    "Mean consecutive trajectory cosine:",
    trajectory_mean
)

print(
    "Mean random-pair cosine:",
    random_mean
)

print(
    "Difference:",
    trajectory_mean
    - random_mean
)

print("=" * 70)