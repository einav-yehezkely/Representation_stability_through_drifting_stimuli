import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    SCRIPT_DIR
)

ROOT_DIR = os.path.dirname(
    PROJECT_DIR
)

FACENET_CSV = os.path.join(
    ROOT_DIR,
    "female_facenet_embeddings.csv"
)

ARCFACE_CSV = os.path.join(
    ROOT_DIR,
    "female_arcface_embeddings.csv"
)

ROTATION_CSV = os.path.join(
    ROOT_DIR,
    "tmp_CE",
    "rotation_sequence_all.csv"
)

OUTPUT_CSV = os.path.join(
    ROOT_DIR,
    "facenet_vs_arcface_path_distances.csv"
)

OUTPUT_GRAPH = os.path.join(
    ROOT_DIR,
    "facenet_vs_arcface_path_distances.png"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(csv_path):

    df = pd.read_csv(
        csv_path,
        header=None
    )

    lookup = {}

    for _, row in df.iterrows():

        filename = os.path.basename(
            str(row.iloc[0]).strip()
        )

        embedding = row.iloc[
            1:513
        ].to_numpy(
            dtype=np.float32
        )

        # L2 normalization
        norm = np.linalg.norm(
            embedding
        )

        if norm > 0:
            embedding = embedding / norm

        lookup[
            filename
        ] = embedding

    return lookup


print("Loading FaceNet embeddings...")

facenet_lookup = load_embeddings(
    FACENET_CSV
)

print(
    "FaceNet embeddings:",
    len(facenet_lookup)
)

print("Loading ArcFace embeddings...")

arcface_lookup = load_embeddings(
    ARCFACE_CSV
)

print(
    "ArcFace embeddings:",
    len(arcface_lookup)
)


# ============================================================
# LOAD ROTATION SEQUENCE
# ============================================================

rotation_df = pd.read_csv(
    ROTATION_CSV
)

filenames = rotation_df[
    "filename"
].astype(str).tolist()


# ============================================================
# DISTANCE FUNCTIONS
# ============================================================

def euclidean_distance(a, b):

    return np.linalg.norm(
        a - b
    )


def cosine_distance(a, b):

    cosine_similarity = np.dot(
        a,
        b
    )

    # Since vectors are already normalized
    return 1 - cosine_similarity


# ============================================================
# COMPARE CONSECUTIVE IMAGES
# ============================================================

records = []

for i in range(
    len(filenames) - 1
):

    f1 = os.path.basename(
        filenames[i]
    )

    f2 = os.path.basename(
        filenames[i + 1]
    )

    if (
        f1 not in facenet_lookup
        or
        f2 not in facenet_lookup
        or
        f1 not in arcface_lookup
        or
        f2 not in arcface_lookup
    ):

        print(
            "Skipping missing pair:",
            f1,
            f2
        )

        continue

    fn1 = facenet_lookup[
        f1
    ]

    fn2 = facenet_lookup[
        f2
    ]

    af1 = arcface_lookup[
        f1
    ]

    af2 = arcface_lookup[
        f2
    ]

    facenet_euclidean = euclidean_distance(
        fn1,
        fn2
    )

    arcface_euclidean = euclidean_distance(
        af1,
        af2
    )

    facenet_cosine = cosine_distance(
        fn1,
        fn2
    )

    arcface_cosine = cosine_distance(
        af1,
        af2
    )

    records.append(
        {
            "step": i,
            "filename_1": f1,
            "filename_2": f2,
            "facenet_euclidean":
                facenet_euclidean,
            "arcface_euclidean":
                arcface_euclidean,
            "facenet_cosine":
                facenet_cosine,
            "arcface_cosine":
                arcface_cosine,
        }
    )


results_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CONSECUTIVE DISTANCE SUMMARY")
print("=" * 70)

print()

print(
    "FaceNet mean Euclidean:",
    results_df[
        "facenet_euclidean"
    ].mean()
)

print(
    "ArcFace mean Euclidean:",
    results_df[
        "arcface_euclidean"
    ].mean()
)

print()

print(
    "FaceNet mean cosine distance:",
    results_df[
        "facenet_cosine"
    ].mean()
)

print(
    "ArcFace mean cosine distance:",
    results_df[
        "arcface_cosine"
    ].mean()
)


# ============================================================
# CORRELATION
# ============================================================

euclidean_corr = results_df[
    [
        "facenet_euclidean",
        "arcface_euclidean"
    ]
].corr().iloc[
    0,
    1
]

cosine_corr = results_df[
    [
        "facenet_cosine",
        "arcface_cosine"
    ]
].corr().iloc[
    0,
    1
]


print()

print(
    "Euclidean distance correlation:",
    euclidean_corr
)

print(
    "Cosine distance correlation:",
    cosine_corr
)

print("=" * 70)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    results_df["step"],
    results_df["facenet_euclidean"],
    label="FaceNet"
)

plt.plot(
    results_df["step"],
    results_df["arcface_euclidean"],
    label="ArcFace"
)

plt.xlabel(
    "Position along rotation sequence"
)

plt.ylabel(
    "Euclidean distance between consecutive images"
)

plt.title(
    "Consecutive Representation Distance Along Rotation Path"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    OUTPUT_GRAPH,
    dpi=300
)

plt.close()


print()
print(
    "Saved CSV:",
    OUTPUT_CSV
)

print(
    "Saved graph:",
    OUTPUT_GRAPH
)