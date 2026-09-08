import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

UMAP_CSV = "shufflenet_umap.csv"
EMBEDDINGS_CSV = "shufflenetSpace/shufflenet_embeddings.csv"

ROTATION_RANGE = 360

RANDOM_PAIRS = 5000
RANDOM_SEED = 42

# Importance of being close to the desired UMAP location
UMAP_WEIGHT = 1.0

# Importance of being similar to the previous image
# in the original ShuffleNet embedding space
EMBEDDING_WEIGHT = 5.0


# ------------------------------------------------------------
# Forward-motion constraint
# ------------------------------------------------------------

# Candidate must move forward by at least this amount
MIN_FORWARD_DEG = 0.0001

# Candidate cannot jump more than this many degrees forward
MAX_FORWARD_DEG = 3.0


OUTPUT_DIR = "shufflenet_smooth_rotation_test"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load UMAP
# ============================================================

umap_df = pd.read_csv(
    UMAP_CSV
)

filename_col = umap_df.columns[0]
x_col = umap_df.columns[1]
y_col = umap_df.columns[2]

umap_df = umap_df[
    [
        filename_col,
        x_col,
        y_col
    ]
].copy()

umap_df.columns = [
    "filename",
    "x",
    "y"
]

umap_df["filename"] = (
    umap_df["filename"]
    .astype(str)
)

names = (
    umap_df["filename"]
    .to_numpy()
)

points = (
    umap_df[
        ["x", "y"]
    ]
    .to_numpy(dtype=float)
)


print(
    "Number of UMAP images:",
    len(names)
)


# ============================================================
# Load original ShuffleNet embeddings
# ============================================================

emb_df = pd.read_csv(
    EMBEDDINGS_CSV
)

embedding_filename_col = (
    emb_df.columns[0]
)

embedding_columns = (
    emb_df.columns[1:]
)

emb_df[
    embedding_filename_col
] = (
    emb_df[
        embedding_filename_col
    ]
    .astype(str)
)


embedding_dict = {}

for _, row in emb_df.iterrows():

    filename = row[
        embedding_filename_col
    ]

    embedding = (
        row[
            embedding_columns
        ]
        .to_numpy(dtype=float)
    )

    embedding_dict[
        filename
    ] = embedding


print(
    "Loaded embeddings:",
    len(embedding_dict)
)

print(
    "Embedding dimension:",
    len(embedding_columns)
)


# ============================================================
# Keep only images appearing in both files
# ============================================================

valid_mask = np.array([
    filename in embedding_dict
    for filename in names
])

names = names[
    valid_mask
]

points = points[
    valid_mask
]


print(
    "Images existing in both files:",
    len(names)
)


if len(names) < 2:

    raise ValueError(
        "Not enough matching images."
    )


# ============================================================
# UMAP centroid
# ============================================================

centroid = (
    points.mean(axis=0)
)

centered = (
    points
    - centroid
)


# ============================================================
# Actual UMAP angle of every image
# ============================================================

actual_angles = (
    np.degrees(
        np.arctan2(
            centered[:, 1],
            centered[:, 0]
        )
    )
    + 360
) % 360


radii = np.linalg.norm(
    centered,
    axis=1
)


# ============================================================
# Number of steps
# ============================================================

NUM_STEPS = min(
    1000,
    len(names)
)

IDEAL_STEP_DEG = (
    ROTATION_RANGE
    / NUM_STEPS
)


print(
    "Number of steps:",
    NUM_STEPS
)

print(
    "Ideal target movement per step:",
    IDEAL_STEP_DEG,
    "degrees"
)


# ============================================================
# Helper functions
# ============================================================

def rotate_vector(
    vector,
    angle_deg
):

    angle_rad = np.deg2rad(
        angle_deg
    )

    rotation_matrix = np.array([
        [
            np.cos(angle_rad),
            -np.sin(angle_rad)
        ],
        [
            np.sin(angle_rad),
            np.cos(angle_rad)
        ]
    ])

    return (
        rotation_matrix
        @ vector
    )


def cosine_similarity(
    a,
    b
):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:

        return np.nan

    return (
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# Choose starting image
# ============================================================

target_radius = (
    np.median(radii)
)


# Find image near 0 degrees
angle_error = np.minimum(
    np.abs(actual_angles),
    360
    - np.abs(actual_angles)
)


radius_error = np.abs(
    radii
    - target_radius
)


start_score = (
    angle_error
    +
    100
    * radius_error
)


base_idx = np.argmin(
    start_score
)

base_point = (
    points[
        base_idx
    ]
)

base_filename = (
    names[
        base_idx
    ]
)

start_actual_angle = (
    actual_angles[
        base_idx
    ]
)


print(
    "\nStarting image:",
    base_filename
)

print(
    "Starting actual UMAP angle:",
    start_actual_angle
)


# ============================================================
# Convert all image angles to forward progress relative
# to the starting image.
#
# Example:
#
# start = 359 degrees
#
# image at 1 degree -> progress = 2 degrees
#
# Therefore wrap-around at 360 does not cause a fake
# backwards movement.
# ============================================================

angular_progress = (
    actual_angles
    - start_actual_angle
) % 360


# ============================================================
# Build smooth FORWARD-ONLY trajectory
# ============================================================

rotation_sequence = []

used_indices = set()

previous_embedding = None

previous_progress = None


for step in range(
    NUM_STEPS
):

    # --------------------------------------------------------
    # Ideal target progress
    # --------------------------------------------------------

    target_progress = (
        ROTATION_RANGE
        * step
        / NUM_STEPS
    )


    # --------------------------------------------------------
    # Ideal target coordinate in UMAP
    # --------------------------------------------------------

    target_point = (
        centroid
        +
        rotate_vector(
            base_point
            - centroid,
            target_progress
        )
    )


    # --------------------------------------------------------
    # Distance from every image to desired UMAP target
    # --------------------------------------------------------

    umap_distances = (
        np.linalg.norm(
            points
            - target_point,
            axis=1
        )
    )


    umap_scale = (
        np.median(
            umap_distances
        )
    )

    if (
        umap_scale == 0
        or not np.isfinite(
            umap_scale
        )
    ):

        umap_scale = 1.0


    normalized_umap = (
        umap_distances
        / umap_scale
    )


    # ========================================================
    # First point
    # ========================================================

    if step == 0:

        best_idx = (
            base_idx
        )

        embedding_distances = (
            np.zeros(
                len(names)
            )
        )

        scores = (
            normalized_umap.copy()
        )


    # ========================================================
    # Every later point
    # ========================================================

    else:

        # ----------------------------------------------------
        # ShuffleNet embedding distance from previous image
        # ----------------------------------------------------

        embedding_distances = (
            np.zeros(
                len(names)
            )
        )


        for idx, filename in enumerate(
            names
        ):

            current_embedding = (
                embedding_dict[
                    filename
                ]
            )

            embedding_distances[
                idx
            ] = np.linalg.norm(
                current_embedding
                - previous_embedding
            )


        embedding_scale = (
            np.median(
                embedding_distances
            )
        )


        if (
            embedding_scale == 0
            or not np.isfinite(
                embedding_scale
            )
        ):

            embedding_scale = 1.0


        normalized_embedding = (
            embedding_distances
            / embedding_scale
        )


        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        scores = (
            UMAP_WEIGHT
            * normalized_umap
            +
            EMBEDDING_WEIGHT
            * normalized_embedding
        )


        # ====================================================
        # CRITICAL PART:
        # only allow FORWARD images
        # ====================================================

        forward_delta = (
            angular_progress
            - previous_progress
        )


        valid_forward = (
            (forward_delta >= MIN_FORWARD_DEG)
            &
            (forward_delta <= MAX_FORWARD_DEG)
        )


        # Reject backwards images
        scores[
            ~valid_forward
        ] = np.inf


        # ----------------------------------------------------
        # Don't reuse images
        # ----------------------------------------------------

        for idx in used_indices:

            scores[
                idx
            ] = np.inf


        # ----------------------------------------------------
        # Find best candidate
        # ----------------------------------------------------

        best_idx = np.argmin(
            scores
        )


        # ====================================================
        # Fallback:
        #
        # If there isn't an image within MAX_FORWARD_DEG,
        # choose the closest UNUSED image that is still
        # forward.
        #
        # It may jump farther, but NEVER backwards.
        # ====================================================

        if not np.isfinite(
            scores[
                best_idx
            ]
        ):

            print(
                f"Step {step}: "
                f"no image within "
                f"{MAX_FORWARD_DEG} degrees. "
                f"Searching farther forward."
            )


            fallback_scores = (
                UMAP_WEIGHT
                * normalized_umap
                +
                EMBEDDING_WEIGHT
                * normalized_embedding
            )


            forward_only = (
                angular_progress
                > previous_progress
            )


            fallback_scores[
                ~forward_only
            ] = np.inf


            for idx in used_indices:

                fallback_scores[
                    idx
                ] = np.inf


            best_idx = np.argmin(
                fallback_scores
            )


            if not np.isfinite(
                fallback_scores[
                    best_idx
                ]
            ):

                print(
                    "No more forward images available."
                )

                break


            scores = (
                fallback_scores
            )


    # ========================================================
    # Selected image
    # ========================================================

    filename = (
        names[
            best_idx
        ]
    )


    current_embedding = (
        embedding_dict[
            filename
        ]
    )


    current_actual_angle = (
        actual_angles[
            best_idx
        ]
    )


    current_progress = (
        angular_progress[
            best_idx
        ]
    )


    # ========================================================
    # Record forward movement
    # ========================================================

    if previous_progress is None:

        forward_step = (
            np.nan
        )

    else:

        forward_step = (
            current_progress
            - previous_progress
        )


    # ========================================================
    # Save row
    # ========================================================

    rotation_sequence.append({

        "step":
            step,

        # IMPORTANT:
        # angle used later by the training script
        # is the ACTUAL angle of the selected image
        "angle":
            current_actual_angle,

        # Ideal target angle/progress
        "target_angle":
            target_progress,

        "actual_progress":
            current_progress,

        "forward_step_deg":
            forward_step,

        "filename":
            filename,

        "umap_x":
            points[
                best_idx,
                0
            ],

        "umap_y":
            points[
                best_idx,
                1
            ],

        "target_x":
            target_point[
                0
            ],

        "target_y":
            target_point[
                1
            ],

        "umap_distance_to_target":
            umap_distances[
                best_idx
            ],

        "embedding_distance_from_previous":
            (
                np.nan
                if previous_embedding is None
                else embedding_distances[
                    best_idx
                ]
            ),

        "combined_score":
            (
                scores[
                    best_idx
                ]
                if np.isfinite(
                    scores[
                        best_idx
                    ]
                )
                else np.nan
            )
    })


    # ========================================================
    # Update state
    # ========================================================

    used_indices.add(
        best_idx
    )


    previous_embedding = (
        current_embedding
    )


    previous_progress = (
        current_progress
    )


# ============================================================
# Save trajectory
# ============================================================

sequence_df = pd.DataFrame(
    rotation_sequence
)


sequence_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "smooth_rotation_sequence.csv"
    ),
    index=False
)


print(
    "\nRotation sequence length:",
    len(sequence_df)
)


# ============================================================
# CHECK: make absolutely sure there are no backwards steps
# ============================================================

if len(sequence_df) > 1:

    backwards = (
        sequence_df[
            "forward_step_deg"
        ]
        .dropna()
        <= 0
    ).sum()


    print(
        "\nBackward steps:",
        backwards
    )


    print(
        "Mean actual forward step:",
        sequence_df[
            "forward_step_deg"
        ]
        .mean()
    )


    print(
        "Median actual forward step:",
        sequence_df[
            "forward_step_deg"
        ]
        .median()
    )


    print(
        "Maximum forward step:",
        sequence_df[
            "forward_step_deg"
        ]
        .max()
    )


# ============================================================
# Consecutive ShuffleNet distances
# ============================================================

consecutive_results = []


for i in range(
    len(sequence_df)
    - 1
):

    row1 = (
        sequence_df.iloc[
            i
        ]
    )

    row2 = (
        sequence_df.iloc[
            i + 1
        ]
    )


    f1 = (
        row1[
            "filename"
        ]
    )

    f2 = (
        row2[
            "filename"
        ]
    )


    z1 = (
        embedding_dict[
            f1
        ]
    )

    z2 = (
        embedding_dict[
            f2
        ]
    )


    euclidean = (
        np.linalg.norm(
            z1 - z2
        )
    )


    cosine = (
        cosine_similarity(
            z1,
            z2
        )
    )


    p1 = np.array([
        row1[
            "umap_x"
        ],
        row1[
            "umap_y"
        ]
    ])


    p2 = np.array([
        row2[
            "umap_x"
        ],
        row2[
            "umap_y"
        ]
    ])


    umap_distance = (
        np.linalg.norm(
            p1 - p2
        )
    )


    consecutive_results.append({

        "step":
            i,

        "image_1":
            f1,

        "image_2":
            f2,

        "forward_step_deg":
            row2[
                "forward_step_deg"
            ],

        "umap_distance":
            umap_distance,

        "embedding_euclidean":
            euclidean,

        "embedding_cosine_similarity":
            cosine
    })


consecutive_df = pd.DataFrame(
    consecutive_results
)


consecutive_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consecutive_distances.csv"
    ),
    index=False
)


# ============================================================
# Random-pair baseline
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


random_results = []


for _ in range(
    RANDOM_PAIRS
):

    idx1, idx2 = (
        rng.choice(
            len(names),
            size=2,
            replace=False
        )
    )


    f1 = (
        names[
            idx1
        ]
    )

    f2 = (
        names[
            idx2
        ]
    )


    z1 = (
        embedding_dict[
            f1
        ]
    )

    z2 = (
        embedding_dict[
            f2
        ]
    )


    euclidean = (
        np.linalg.norm(
            z1 - z2
        )
    )


    cosine = (
        cosine_similarity(
            z1,
            z2
        )
    )


    random_results.append({

        "image_1":
            f1,

        "image_2":
            f2,

        "embedding_euclidean":
            euclidean,

        "embedding_cosine_similarity":
            cosine
    })


random_df = pd.DataFrame(
    random_results
)


random_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_distances.csv"
    ),
    index=False
)


# ============================================================
# Statistics
# ============================================================

consecutive_euc = (
    consecutive_df[
        "embedding_euclidean"
    ]
)

random_euc = (
    random_df[
        "embedding_euclidean"
    ]
)


consecutive_cos = (
    consecutive_df[
        "embedding_cosine_similarity"
    ]
)

random_cos = (
    random_df[
        "embedding_cosine_similarity"
    ]
)


print(
    "\n========================"
)

print(
    "RESULTS"
)

print(
    "========================"
)


print(
    "\nEuclidean distance:"
)


print(
    "Consecutive mean:",
    consecutive_euc.mean()
)


print(
    "Consecutive median:",
    consecutive_euc.median()
)


print(
    "Random mean:",
    random_euc.mean()
)


print(
    "Random median:",
    random_euc.median()
)


ratio = (
    consecutive_euc.mean()
    /
    random_euc.mean()
)


print(
    "\nConsecutive / random ratio:",
    ratio
)


print(
    "\nCosine similarity:"
)


print(
    "Consecutive mean:",
    consecutive_cos.mean()
)


print(
    "Consecutive median:",
    consecutive_cos.median()
)


print(
    "Random mean:",
    random_cos.mean()
)


print(
    "Random median:",
    random_cos.median()
)


fraction_below_random_mean = (
    consecutive_euc
    <
    random_euc.mean()
).mean()


print(
    "\nFraction of consecutive pairs "
    "closer than average random pair:",
    fraction_below_random_mean
)


# ============================================================
# Plot 1 - Embedding-distance histogram
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.hist(
    random_euc,
    bins=50,
    density=True,
    alpha=0.5,
    label="Random pairs"
)


plt.hist(
    consecutive_euc,
    bins=50,
    density=True,
    alpha=0.7,
    label="Consecutive forward images"
)


plt.xlabel(
    "Euclidean distance in ShuffleNet embedding"
)


plt.ylabel(
    "Density"
)


plt.title(
    "Forward smooth trajectory vs random pairs"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "embedding_distance_histogram.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# Plot 2 - Cosine similarity
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.hist(
    random_cos.dropna(),
    bins=50,
    density=True,
    alpha=0.5,
    label="Random pairs"
)


plt.hist(
    consecutive_cos.dropna(),
    bins=50,
    density=True,
    alpha=0.7,
    label="Consecutive forward images"
)


plt.xlabel(
    "Cosine similarity"
)


plt.ylabel(
    "Density"
)


plt.title(
    "ShuffleNet similarity along forward trajectory"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cosine_similarity_histogram.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# Plot 3 - Distance along trajectory
# ============================================================

plt.figure(
    figsize=(11, 5)
)


plt.plot(
    consecutive_df[
        "step"
    ],
    consecutive_euc
)


plt.axhline(
    random_euc.mean(),
    linestyle="--",
    label="Random-pair mean"
)


plt.xlabel(
    "Rotation step"
)


plt.ylabel(
    "ShuffleNet distance to next image"
)


plt.title(
    "Representation continuity along forward trajectory"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "distance_along_rotation.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# Plot 4 - Actual angular progress
#
# This is the important new graph.
# It should NEVER go down.
# ============================================================

plt.figure(
    figsize=(11, 5)
)


plt.plot(
    sequence_df[
        "step"
    ],
    sequence_df[
        "actual_progress"
    ],
    label="Selected image"
)


plt.plot(
    sequence_df[
        "step"
    ],
    sequence_df[
        "target_angle"
    ],
    linestyle="--",
    label="Ideal target"
)


plt.xlabel(
    "Step"
)


plt.ylabel(
    "Angular progress (degrees)"
)


plt.title(
    "Forward angular progression"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "angular_progress.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# Plot 5 - Actual UMAP trajectory
# ============================================================

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    points[:, 0],
    points[:, 1],
    s=5,
    alpha=0.2,
    label="All images"
)


plt.plot(
    sequence_df[
        "umap_x"
    ],
    sequence_df[
        "umap_y"
    ],
    linewidth=1,
    label="Selected trajectory"
)


plt.scatter(
    sequence_df[
        "umap_x"
    ],
    sequence_df[
        "umap_y"
    ],
    s=10
)


plt.xlabel(
    "UMAP 1"
)


plt.ylabel(
    "UMAP 2"
)


plt.title(
    "Forward smooth UMAP trajectory"
)


plt.axis(
    "equal"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "smooth_rotation_umap.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# Plot 6 - Ideal target vs selected images
# ============================================================

plt.figure(
    figsize=(8, 8)
)


plt.plot(
    sequence_df[
        "target_x"
    ],
    sequence_df[
        "target_y"
    ],
    label="Ideal rotating target"
)


plt.plot(
    sequence_df[
        "umap_x"
    ],
    sequence_df[
        "umap_y"
    ],
    label="Selected images"
)


plt.xlabel(
    "UMAP 1"
)


plt.ylabel(
    "UMAP 2"
)


plt.title(
    "Ideal rotation vs forward selected trajectory"
)


plt.axis(
    "equal"
)


plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "target_vs_selected_trajectory.png"
    ),
    dpi=200
)


plt.close()


print(
    "\nResults saved to:",
    OUTPUT_DIR
)


print(
    "\nUMAP_WEIGHT =",
    UMAP_WEIGHT
)


print(
    "EMBEDDING_WEIGHT =",
    EMBEDDING_WEIGHT
)


print(
    "Maximum allowed normal forward step =",
    MAX_FORWARD_DEG,
    "degrees"
)