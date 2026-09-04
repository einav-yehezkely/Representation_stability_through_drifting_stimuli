import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# SETTINGS
# ============================================================

EMBEDDINGS_CSV = "female_arcface_embeddings.csv"
DATA_DIR = "split_data"


# ============================================================
# LOAD ARCFACE EMBEDDINGS
# ============================================================

print("=" * 70)
print("LOADING ARCFACE EMBEDDINGS")
print("=" * 70)

embeddings_df = pd.read_csv(
    EMBEDDINGS_CSV,
    header=None,
)

if embeddings_df.shape[1] != 513:
    raise RuntimeError(
        f"Expected 513 columns, found {embeddings_df.shape[1]}"
    )


embedding_lookup = {}

for _, row in embeddings_df.iterrows():

    filename = str(row.iloc[0]).strip()
    basename = os.path.basename(filename)

    vector = row.iloc[1:513].to_numpy(
        dtype=np.float32
    )

    # Same normalization as current ArcFace classifier
    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    embedding_lookup[filename] = vector
    embedding_lookup[basename] = vector


print("Embeddings loaded:", len(embeddings_df))


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):

    X = []
    y = []
    filenames = []

    split_dir = os.path.join(
        DATA_DIR,
        split_name,
    )

    for label, class_name in enumerate(["A", "B"]):

        class_dir = os.path.join(
            split_dir,
            class_name,
        )

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(class_dir)

        for filename in sorted(os.listdir(class_dir)):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            basename = os.path.basename(filename)

            if basename not in embedding_lookup:
                print(
                    "WARNING: missing embedding:",
                    basename
                )
                continue

            X.append(
                embedding_lookup[basename]
            )

            y.append(label)
            filenames.append(basename)

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    filenames = np.asarray(filenames)

    return X, y, filenames


X_train, y_train, train_files = load_split("train")
X_val, y_val, val_files = load_split("val")


print()
print("=" * 70)
print("DATA")
print("=" * 70)

print("Train:", X_train.shape)
print("Val:", X_val.shape)

print()

print(
    "Train A:",
    np.sum(y_train == 0)
)

print(
    "Train B:",
    np.sum(y_train == 1)
)

print(
    "Val A:",
    np.sum(y_val == 0)
)

print(
    "Val B:",
    np.sum(y_val == 1)
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train,
    )

    train_pred = model.predict(
        X_train
    )

    val_pred = model.predict(
        X_val
    )

    train_acc = accuracy_score(
        y_train,
        train_pred,
    )

    val_acc = accuracy_score(
        y_val,
        val_pred,
    )

    print(
        f"TRAIN accuracy = "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"VAL accuracy   = "
        f"{val_acc * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("Confusion matrix:")

    cm = confusion_matrix(
        y_val,
        val_pred,
    )

    print(cm)


    print()
    print("Classification report:")

    print(
        classification_report(
            y_val,
            val_pred,
            target_names=["A", "B"],
            digits=4,
        )
    )


    # --------------------------------------------------------
    # Misclassified images
    # --------------------------------------------------------

    wrong = np.where(
        val_pred != y_val
    )[0]

    print()
    print(
        "Number of misclassified validation images:",
        len(wrong),
    )

    if len(wrong) > 0:

        wrong_rows = []

        for idx in wrong:

            true_label = (
                "A"
                if y_val[idx] == 0
                else "B"
            )

            predicted_label = (
                "A"
                if val_pred[idx] == 0
                else "B"
            )

            wrong_rows.append(
                {
                    "filename": val_files[idx],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                }
            )

        wrong_df = pd.DataFrame(
            wrong_rows
        )

        output_name = (
            name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_misclassified.csv"
        )

        wrong_df.to_csv(
            output_name,
            index=False,
        )

        print()
        print("Misclassified images:")

        print(
            wrong_df.to_string(
                index=False
            )
        )

        print()
        print(
            "Saved:",
            output_name
        )

    return val_acc


# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================

logistic_model = LogisticRegression(
    max_iter=5000,
    C=1.0,
)

logistic_acc = evaluate_model(
    "Logistic Regression",
    logistic_model,
)


# ============================================================
# 2. LINEAR SVM
# ============================================================

linear_svm = SVC(
    kernel="linear",
    C=1.0,
)

linear_acc = evaluate_model(
    "Linear SVM",
    linear_svm,
)


# ============================================================
# 3. RBF SVM
# ============================================================

rbf_svm = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
)

rbf_acc = evaluate_model(
    "RBF SVM",
    rbf_svm,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Logistic Regression : "
    f"{logistic_acc * 100:.2f}%"
)

print(
    f"Linear SVM          : "
    f"{linear_acc * 100:.2f}%"
)

print(
    f"RBF SVM             : "
    f"{rbf_acc * 100:.2f}%"
)

print("=" * 70)


best_acc = max(
    logistic_acc,
    linear_acc,
    rbf_acc,
)

if best_acc >= 0.99:

    print()
    print(
        "At least one classifier reached >= 99%."
    )

    print(
        "The ArcFace representation appears capable "
        "of separating A/B at the desired level."
    )

else:

    print()
    print(
        "No classifier reached 99%."
    )

    print(
        "This suggests the limitation may be in the "
        "A/B separability of the ArcFace representation, "
        "rather than only in the MLP training."
    )