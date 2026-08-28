import os
import math
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

# True  -> supervised:
#          use the real A/B label
#
# False -> unsupervised / self-learning:
#          use the model's own prediction as pseudo-label
SUPERVISED = True


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

IMAGE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "female_faces"
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ROTATION_CSV = os.path.join(
    PROJECT_ROOT,
    "shufflenet_smooth_rotation_test",
    "smooth_rotation_sequence.csv"
)

ALL_IMAGES_UMAP_CSV = os.path.join(
    PROJECT_ROOT,
    "shufflenet_umap.csv"
)

rotation_df = pd.read_csv(
    ROTATION_CSV
)

all_images_df = pd.read_csv(
    ALL_IMAGES_UMAP_CSV
)

print(
    "Number of images available for local batches:",
    len(all_images_df)
)

TRAINED_MODEL = r"model_ft_0_CE_UMAP_last_layer.pth"

ANGLE_EVERY = 25

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

LEARNING_RATE = 0.001

NUM_ROTATIONS = 1

SEED = 42

IMAGES_PER_STEP = 50


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_CSV = (
    "matlab_like_supervised.csv"
    if SUPERVISED
    else "matlab_like_unsupervised.csv"
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("MATLAB-LIKE ROTATION EXPERIMENT")
print("=" * 80)

print("Device:", DEVICE)

if SUPERVISED:
    print("Mode: SUPERVISED")
else:
    print("Mode: UNSUPERVISED / SELF-LEARNING")

print("Learning rate:", LEARNING_RATE)
print("Model:", TRAINED_MODEL)
print("Rotation CSV:", ROTATION_CSV)
print()


# ============================================================
# IMAGE PREPROCESSING
#
# Must be the same preprocessing that was used when the
# original ShuffleNet classifier was trained.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TRAINED SHUFFLENET
# ============================================================

print("=" * 80)
print("LOADING TRAINED MODEL")
print("=" * 80)


model = models.shufflenet_v2_x0_5(
    weights=None
)

num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 2),
)


# ============================================================
# LOAD CHECKPOINT
#
# Supports both:
#
# torch.save(model.state_dict(), ...)
#
# and:
#
# torch.save({
#     "model_state_dict": model.state_dict(),
#     ...
# }, ...)
# ============================================================

checkpoint = torch.load(
    TRAINED_MODEL,
    map_location=DEVICE
)


if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(state_dict)

model = model.to(DEVICE)


print("Trained model loaded successfully.")
print()


# ============================================================
# FREEZE EVERYTHING EXCEPT FC
#
# MATLAB analogy:
#
# ShuffleNet representation = fixed input representation
# fc weights              = theta
#
# Only theta changes.
# ============================================================

for param in model.parameters():
    param.requires_grad = False


for param in model.fc.parameters():
    param.requires_grad = True


# IMPORTANT:
#
# Keep the CNN in eval mode.
#
# Gradients still work for fc in eval mode.
# This prevents BatchNorm running statistics from changing.
#
# We want ONLY fc to change.
model.eval()


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"Trainable parameters: "
    f"{trainable_parameters:,} / {total_parameters:,}"
)

print("Only model.fc will be updated.")
print()


# ============================================================
# OPTIMIZER
#
# Simple SGD:
#
# theta <- theta - learning_rate * gradient
#
# No Adam
# No AdamW
# No momentum
# No scheduler
# No weight decay
# ============================================================

# optimizer = optim.SGD(
#     model.fc.parameters(),
#     lr=LEARNING_RATE
# )
optimizer = optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# LOAD ROTATION SEQUENCE
# ============================================================

print("=" * 80)
print("LOADING ROTATION SEQUENCE")
print("=" * 80)


rotation_df = pd.read_csv(
    ROTATION_CSV
)


print("Columns:", list(rotation_df.columns))
print("Number of rows:", len(rotation_df))
print()


# ============================================================
# FIND FILENAME COLUMN
# ============================================================

possible_filename_columns = [
    "filename",
    "file_name",
    "image",
    "image_name",
    "name"
]


filename_column = None


for column in possible_filename_columns:

    if column in rotation_df.columns:

        filename_column = column
        break


if filename_column is None:

    raise ValueError(
        "\nCould not find a filename column.\n"
        f"CSV columns are:\n{list(rotation_df.columns)}"
    )


print("Filename column:", filename_column)


# ============================================================
# FIND LABEL COLUMN
#
# Only necessary in supervised mode.
# ============================================================




# ============================================================
# LABEL CONVERSION
#
# A -> 0
# B -> 1
# ============================================================

def convert_label(value):

    if isinstance(value, (int, np.integer)):

        value = int(value)

        if value in [0, 1]:
            return value


    if isinstance(value, (float, np.floating)):

        value = int(value)

        if value in [0, 1]:
            return value


    value = str(value).strip().upper()


    if value == "A":
        return 0

    if value == "B":
        return 1

    if value == "0":
        return 0

    if value == "1":
        return 1


    raise ValueError(
        f"Unknown label: {value}. "
        "Expected A/B or 0/1."
    )


# ============================================================
# CLASS NAME
# ============================================================

def class_name(label):

    if label == 0:
        return "A"

    return "B"


# ============================================================
# LOAD ONE IMAGE
# ============================================================

def load_image(image_name):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )


    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )


    image = Image.open(
        image_path
    ).convert("RGB")


    image = transform(image)


    # [C,H,W]
    # ->
    # [1,C,H,W]

    image = image.unsqueeze(0)


    return image.to(DEVICE)


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

@torch.no_grad()
def predict(image):

    model.eval()


    logits = model(image)


    probabilities = torch.softmax(
        logits,
        dim=1
    )


    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


    pA = probabilities[0, 0].item()
    pB = probabilities[0, 1].item()


    return (
        predicted_class,
        pA,
        pB
    )


# ============================================================
# ONE MATLAB-LIKE STEP
#
#
# MATLAB:
#
# x
# ↓
# prediction
# ↓
#
# supervised:
#     y = correct label
#
# unsupervised:
#     y = current prediction
#
# ↓
# ONE gradient step
#
#
# Python:
#
# one image
# ↓
# trained ShuffleNet
# ↓
# prediction
# ↓
# choose label
# ↓
# ONE SGD step on fc
#
# ============================================================

def train_one_step(
    image,
    true_label=None
):

    # ========================================================
    # 1. PREDICTION BEFORE TRAINING
    # ========================================================

    (
        predicted_before,
        pA_before,
        pB_before

    ) = predict(image)


    # ========================================================
    # 2. CHOOSE TRAINING LABEL
    # ========================================================

    if SUPERVISED:

        if true_label is None:

            raise ValueError(
                "Supervised mode requires a true label."
            )


        training_label = true_label


    else:

        # ----------------------------------------------------
        # SELF-LEARNING
        #
        # The model's OWN current prediction becomes the label.
        # ----------------------------------------------------

        training_label = predicted_before


    # ========================================================
    # 3. CREATE TARGET
    # ========================================================

    target = torch.tensor(
        [training_label],
        dtype=torch.long,
        device=DEVICE
    )


    # ========================================================
    # 4. ONE SGD STEP
    # ========================================================

    # Keep model in eval mode!
    #
    # This freezes BatchNorm behavior.
    #
    # Gradients for fc still work normally.

    model.eval()


    optimizer.zero_grad()


    logits = model(image)


    loss = criterion(
        logits,
        target
    )


    loss.backward()


    optimizer.step()


    # ========================================================
    # 5. PREDICTION AFTER TRAINING
    # ========================================================

    (
        predicted_after,
        pA_after,
        pB_after

    ) = predict(image)


    return {

        "predicted_before":
            predicted_before,

        "training_label":
            training_label,

        "predicted_after":
            predicted_after,

        "pA_before":
            pA_before,

        "pB_before":
            pB_before,

        "pA_after":
            pA_after,

        "pB_after":
            pB_after,

        "loss":
            loss.item()
    }


# ============================================================
# CLASSIFIER WEIGHT VECTOR
#
# For a two-class linear classifier:
#
# score_A = wA*x + bA
# score_B = wB*x + bB
#
# Decision boundary is determined by:
#
# wB - wA
#
# ============================================================

@torch.no_grad()
def get_classifier_vector():

    # Last Linear layer in:
    #
    # Dropout
    # Linear(1024 -> 256)
    # ReLU
    # Dropout
    # Linear(256 -> 2)
    #
    weights = model.fc[4].weight.detach()

    wA = weights[0]
    wB = weights[1]

    return (wB - wA).clone()


# ============================================================
# INITIAL CLASSIFIER VECTOR
# ============================================================

previous_weight_vector = (
    get_classifier_vector()
)

def get_nearby_images(row_index, k=IMAGES_PER_STEP):

    current_row = rotation_df.iloc[row_index]

    x0 = float(current_row["umap_x"])
    y0 = float(current_row["umap_y"])

    distances = (
        (all_images_df["UMAP1"].astype(float) - x0) ** 2
        +
        (all_images_df["UMAP2"].astype(float) - y0) ** 2
    )

    nearest_indices = distances.nsmallest(k).index

    images = []
    filenames = []

    for idx in nearest_indices:

        image_name = str(
            all_images_df.loc[idx, "filename"]
        )

        images.append(
            load_image(image_name)
        )

        filenames.append(image_name)

    return images, filenames
# ============================================================
# INITIAL MODEL PERFORMANCE
#
# Useful sanity check:
# the loaded model should already know A/B.
# ============================================================

print("=" * 80)
print("INITIAL MODEL")
print("=" * 80)

print(
    "FC weight norm:",
    torch.norm(
        previous_weight_vector
    ).item()
)

print()

# ============================================================
# ESTIMATE CLASSIFIER ANGLE IN UMAP SPACE
#
# We cannot directly calculate an angle from model.fc because
# its weights are high-dimensional.
#
# Instead:
#
# 1. Run the current model on all images in the UMAP path.
# 2. For each image calculate:
#
#       score = logit_B - logit_A
#
# 3. Fit:
#
#       score ≈ a * UMAP_x + b * UMAP_y + c
#
# 4. The decision boundary is:
#
#       a*x + b*y + c = 0
#
# Its normal vector is (a,b), so we can calculate its angle.
# ============================================================

@torch.no_grad()
def get_umap_classifier_angle():

    model.eval()

    xs = []
    ys = []
    scores = []

    for _, row in rotation_df.iterrows():

        image_name = str(row[filename_column])

        image = load_image(image_name)

        logits = model(image)

        # Positive -> B
        # Negative -> A
        score = (
            logits[0, 1] - logits[0, 0]
        ).item()

        xs.append(float(row["umap_x"]))
        ys.append(float(row["umap_y"]))
        scores.append(score)


    X = np.column_stack([
        xs,
        ys,
        np.ones(len(xs))
    ])

    y = np.array(scores)


    # Least-squares fit:
    #
    # score = a*x + b*y + c
    #
    coefficients, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    a = coefficients[0]
    b = coefficients[1]


    # Normal-vector angle
    normal_angle = np.degrees(
        np.arctan2(b, a)
    )


    # Convert to 0..360
    normal_angle = normal_angle % 360


    return normal_angle

def train_one_batch(images, true_label=None):

    images = torch.cat(images, dim=0)

    # prediction before
    model.eval()

    with torch.no_grad():

        logits_before = model(images)

        probs_before = torch.softmax(
            logits_before,
            dim=1
        )

        preds_before = logits_before.argmax(dim=1)

        mean_pA_before = probs_before[:, 0].mean().item()
        mean_pB_before = probs_before[:, 1].mean().item()

        majority_pred_before = (
            preds_before.float().mean() >= 0.5
        )

        predicted_before = int(
            majority_pred_before
        )


    # labels
    if SUPERVISED:

        if true_label is None:
            raise ValueError(
                "Supervised mode requires true_label"
            )

        labels = torch.full(
            (images.shape[0],),
            true_label,
            dtype=torch.long,
            device=DEVICE
        )

        training_label = true_label

    else:

        labels = preds_before.detach()

        training_label = predicted_before


    # one optimization step
    optimizer.zero_grad()

    logits = model(images)

    loss = criterion(
        logits,
        labels
    )

    loss.backward()

    optimizer.step()


    # prediction after
    with torch.no_grad():

        logits_after = model(images)

        probs_after = torch.softmax(
            logits_after,
            dim=1
        )

        preds_after = logits_after.argmax(dim=1)

        mean_pA_after = probs_after[:, 0].mean().item()
        mean_pB_after = probs_after[:, 1].mean().item()

        predicted_after = int(
            preds_after.float().mean() >= 0.5
        )


    return {
        "predicted_before": predicted_before,
        "training_label": training_label,
        "predicted_after": predicted_after,

        "pA_before": mean_pA_before,
        "pB_before": mean_pB_before,

        "pA_after": mean_pA_after,
        "pB_after": mean_pB_after,

        "loss": loss.item()
    }
# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 80)
print("STARTING ROTATION")
print("=" * 80)


if SUPERVISED:

    print(
        "Each image is trained using "
        "its TRUE label."
    )

else:

    print(
        "Each image is trained using "
        "the model's OWN prediction."
    )


print()
print(
    "One image -> one label -> "
    "one SGD step -> next image"
)

print()


results = []

iteration = 0

example_angles = []
classifier_angles = []

# ============================================================
# ROTATIONS
# ============================================================

for rotation in range(NUM_ROTATIONS):

    print()
    print(
        "=" * 30,
        f"ROTATION {rotation + 1}/{NUM_ROTATIONS}",
        "=" * 30
    )

    print()


    # ========================================================
    # ONE IMAGE AT A TIME
    # ========================================================

    for row_index, row in rotation_df.iterrows():

        image_name = str(
            row[filename_column]
        )


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        # ----------------------------------------------------
        # True label
        #
        # Only used in supervised mode.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # True label
        # ----------------------------------------------------

        if SUPERVISED:
            true_label = 0
        else:
            true_label = None


        # ----------------------------------------------------
        # Get local group around current path point
        # ----------------------------------------------------

        images, batch_filenames = get_nearby_images(
            row_index,
            k=IMAGES_PER_STEP
        )


        # ----------------------------------------------------
        # ONE gradient step on the whole local group
        # ----------------------------------------------------

        result = train_one_batch(
            images=images,
            true_label=true_label
        )

        # ----------------------------------------------------
        # ONE MATLAB-LIKE TRAINING STEP
        # ----------------------------------------------------


        # ============================================================
        # MATLAB-LIKE ANGLES
        # ============================================================

        example_angle = float(row["angle"]) % 360
        example_angles.append(example_angle)

        if iteration % ANGLE_EVERY == 0:
            classifier_angle = get_umap_classifier_angle()
        else:
            classifier_angle = np.nan

        classifier_angles.append(classifier_angle)


        # ====================================================
        # MEASURE HOW MUCH FC CHANGED
        # ====================================================

        current_weight_vector = (
            get_classifier_vector()
        )


        # ----------------------------------------------------
        # Euclidean change
        # ----------------------------------------------------

        weight_change_norm = torch.norm(
            current_weight_vector
            - previous_weight_vector
        ).item()


        # ----------------------------------------------------
        # Angular change between previous and new fc
        # ----------------------------------------------------

        previous_norm = torch.norm(
            previous_weight_vector
        ).item()

        current_norm = torch.norm(
            current_weight_vector
        ).item()


        if (
            previous_norm > 0
            and current_norm > 0
        ):

            cosine = (
                torch.dot(
                    previous_weight_vector,
                    current_weight_vector
                )
                /
                (
                    previous_norm
                    * current_norm
                )
            ).item()


            cosine = max(
                -1.0,
                min(1.0, cosine)
            )


            weight_change_angle = (
                math.degrees(
                    math.acos(cosine)
                )
            )


        else:

            weight_change_angle = 0.0


        previous_weight_vector = (
            current_weight_vector.clone()
        )


        # ====================================================
        # NAMES
        # ====================================================

        pred_before_name = class_name(
            result["predicted_before"]
        )


        train_label_name = class_name(
            result["training_label"]
        )


        pred_after_name = class_name(
            result["predicted_after"]
        )


        if true_label is not None:

            true_label_name = class_name(
                true_label
            )

        else:

            true_label_name = ""


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"{iteration:4d} | "
            f"{image_name:15s} | "
            f"pred={pred_before_name} | "
            f"train={train_label_name} | "
            f"P(A)={result['pA_before']:.4f} | "
            f"P(B)={result['pB_before']:.4f} | "
            f"loss={result['loss']:.6f} | "
            f"dW={weight_change_norm:.6f} | "
            f"dAngle={weight_change_angle:.6f}°"
        )


        # ====================================================
        # RESULT ROW
        # ====================================================

        output_row = {

            "iteration":
                iteration,

            "rotation":
                rotation,

            "filename":
                image_name,

            "true_label":
                true_label_name,

            "predicted_before":
                pred_before_name,

            "training_label":
                train_label_name,

            "predicted_after":
                pred_after_name,

            "pA_before":
                result["pA_before"],

            "pB_before":
                result["pB_before"],

            "pA_after":
                result["pA_after"],

            "pB_after":
                result["pB_after"],

            "loss":
                result["loss"],

            "weight_change_norm":
                weight_change_norm,

            "weight_change_angle_deg":
                weight_change_angle
        }


        # ====================================================
        # COPY OTHER COLUMNS FROM THE ROTATION CSV
        #
        # For example:
        # angle
        # x
        # y
        # etc.
        # ====================================================

        for column in rotation_df.columns:

            if column not in output_row:

                output_row[column] = row[column]


        results.append(
            output_row
        )


        iteration += 1


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 80)
print("FINISHED")
print("=" * 80)


print(
    "Mode:",
    "SUPERVISED"
    if SUPERVISED
    else "UNSUPERVISED"
)


print(
    "Total SGD steps:",
    iteration
)


print(
    "Results saved to:",
    OUTPUT_CSV
)


print()
print(
    "Only model.fc was modified. "
    "The ShuffleNet feature extractor remained frozen."
)

import matplotlib.pyplot as plt


# ============================================================
# MATLAB-LIKE ANGLE GRAPH
# ============================================================

example_angles = np.array(
    example_angles
)

classifier_angles = np.array(
    classifier_angles
)


# Fill the angles that were not calculated
classifier_series = pd.Series(
    classifier_angles
)

classifier_angles_interp = (
    classifier_series
    .interpolate()
    .bfill()
    .ffill()
    .to_numpy()
)


# ------------------------------------------------------------
# Unwrap example angle
# ------------------------------------------------------------

example_angles_unwrapped = np.degrees(
    np.unwrap(
        np.radians(example_angles)
    )
)


# ------------------------------------------------------------
# Unwrap classifier angle
# ------------------------------------------------------------

classifier_angles_unwrapped = np.degrees(
    np.unwrap(
        np.radians(classifier_angles_interp)
    )
)


# ------------------------------------------------------------
# Resolve 180-degree ambiguity
# ------------------------------------------------------------

while (
    classifier_angles_unwrapped[0]
    - example_angles_unwrapped[0]
    > 90
):
    classifier_angles_unwrapped -= 180


while (
    classifier_angles_unwrapped[0]
    - example_angles_unwrapped[0]
    < -90
):
    classifier_angles_unwrapped += 180


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    range(len(example_angles_unwrapped)),
    example_angles_unwrapped,
    label="Examples"
)

plt.plot(
    range(len(classifier_angles_unwrapped)),
    classifier_angles_unwrapped,
    label="Classifier"
)

plt.xlabel("Trial")
plt.ylabel("Angle (degrees)")

plt.title(
    "Examples vs classifier angle"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "matlab_like_angle_tracking.png",
    dpi=150
)

plt.close()

print(
    "Saved MATLAB-like graph: "
    "matlab_like_angle_tracking.png"
)