import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# SETTINGS
# ============================================================

EMBEDDINGS_CSV = "female_arcface_embeddings.csv"
MODEL_PATH = "model_ft_0_ARCFACE_RESNET50.pth"

DATA_DIR = "split_data"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print()
print("=" * 70)
print("LOADING ARCFACE EMBEDDINGS")
print("=" * 70)

df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None
)

print("CSV shape:", df.shape)

if df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns "
        f"(filename + 512 embedding values), "
        f"got {df.shape[1]}"
    )


embedding_lookup = {}

for _, row in df.iterrows():

    filename = str(
        row.iloc[0]
    ).strip()

    basename = os.path.basename(
        filename
    )

    embedding = row.iloc[
        1:513
    ].to_numpy(
        dtype=np.float32
    )

    # Same normalization as in the classifier training code
    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    embedding_lookup[basename] = embedding


print(
    "Loaded embeddings:",
    len(embedding_lookup)
)


# ============================================================
# MODEL
# IMPORTANT:
# This architecture MUST match the trained model.
# ============================================================

class ArcFaceClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2
            )
        )


    def forward(
        self,
        x
    ):

        return self.classifier(
            x
        )


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = ArcFaceClassifier().to(
    device
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print()
print(
    "Loaded model:",
    MODEL_PATH
)


# ============================================================
# EVALUATE ONE FOLDER
# ============================================================

@torch.no_grad()
def evaluate_folder(
    folder,
    true_label,
    class_name
):

    filenames = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    total = 0
    correct = 0

    probabilities_A = []
    probabilities_B = []

    results = []

    missing = []


    for filename in filenames:

        basename = os.path.basename(
            filename
        )

        if basename not in embedding_lookup:

            missing.append(
                basename
            )

            continue


        embedding = embedding_lookup[
            basename
        ]

        x = torch.tensor(
            embedding,
            dtype=torch.float32
        ).unsqueeze(0).to(
            device
        )


        # ---------------------------------------------
        # MODEL OUTPUT
        # ---------------------------------------------

        logits = model(
            x
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )


        p_A = probabilities[
            0,
            0
        ].item()

        p_B = probabilities[
            0,
            1
        ].item()


        predicted_label = torch.argmax(
            probabilities,
            dim=1
        ).item()


        total += 1

        if predicted_label == true_label:
            correct += 1


        probabilities_A.append(
            p_A
        )

        probabilities_B.append(
            p_B
        )


        results.append({
            "filename": basename,
            "true_class": class_name,
            "predicted_class":
                "A"
                if predicted_label == 0
                else "B",
            "P_A": p_A,
            "P_B": p_B
        })


    # ========================================================
    # SUMMARY
    # ========================================================

    accuracy = (
        correct / total
        if total > 0
        else 0
    )


    print()
    print("=" * 70)

    print(
        f"CLASS {class_name}"
    )

    print("=" * 70)

    print(
        "Images:",
        total
    )

    print(
        "Correct:",
        correct
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        f"Mean P(A): "
        f"{np.mean(probabilities_A):.4f}"
    )

    print(
        f"Mean P(B): "
        f"{np.mean(probabilities_B):.4f}"
    )


    if class_name == "A":

        print(
            f"Mean probability of "
            f"CORRECT class: "
            f"{np.mean(probabilities_A):.4f}"
        )

    else:

        print(
            f"Mean probability of "
            f"CORRECT class: "
            f"{np.mean(probabilities_B):.4f}"
        )


    # --------------------------------------------------------
    # Lowest confidence examples
    # --------------------------------------------------------

    if class_name == "A":

        results_sorted = sorted(
            results,
            key=lambda x: x["P_A"]
        )

    else:

        results_sorted = sorted(
            results,
            key=lambda x: x["P_B"]
        )


    print()
    print(
        "10 lowest-confidence examples:"
    )

    for r in results_sorted[:10]:

        print(
            r["filename"],
            "| true:",
            r["true_class"],
            "| predicted:",
            r["predicted_class"],
            "| P(A):",
            f'{r["P_A"]:.4f}',
            "| P(B):",
            f'{r["P_B"]:.4f}'
        )


    if missing:

        print()
        print(
            "Missing embeddings:",
            len(missing)
        )

        print(
            "Examples:",
            missing[:10]
        )


    return pd.DataFrame(
        results
    )


# ============================================================
# TEST TRAIN A
# ============================================================

folder_A = os.path.join(
    DATA_DIR,
    "train",
    "A"
)

results_A = evaluate_folder(
    folder=folder_A,
    true_label=0,
    class_name="A"
)


# ============================================================
# TEST TRAIN B
# ============================================================

folder_B = os.path.join(
    DATA_DIR,
    "train",
    "B"
)

results_B = evaluate_folder(
    folder=folder_B,
    true_label=1,
    class_name="B"
)


# ============================================================
# COMBINE + SAVE RESULTS
# ============================================================

all_results = pd.concat(
    [
        results_A,
        results_B
    ],
    ignore_index=True
)

all_results.to_csv(
    "arcface_initial_AB_classification_check.csv",
    index=False
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_correct = (
    all_results[
        "true_class"
    ]
    ==
    all_results[
        "predicted_class"
    ]
).mean()


print()
print("=" * 70)
print("OVERALL INITIAL A/B CLASSIFICATION")
print("=" * 70)

print(
    f"Overall accuracy: "
    f"{overall_correct * 100:.2f}%"
)

print()

print(
    "Saved detailed results to:"
)

print(
    "arcface_initial_AB_classification_check.csv"
)

print("=" * 70)