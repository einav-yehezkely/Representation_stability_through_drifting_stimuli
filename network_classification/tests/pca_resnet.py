import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

PCA_CSV = "pca_top2_filtered_female_vgg_resnet.csv"
SPLIT_DIR = "split_data"

OUTPUT_PATH = "resnet_pca_split_data_colored.png"


# ============================================================
# LOAD PCA
#
# Expected columns:
# filename, PC1, PC2
# ============================================================

df = pd.read_csv(
    PCA_CSV,
    header=None,
    names=["filename", "PC1", "PC2"]
)

df["filename"] = (
    df["filename"]
    .astype(str)
    .apply(os.path.basename)
    .str.strip()
)


# ============================================================
# GET FILENAMES FROM split_data
# ============================================================

def get_filenames(class_name):

    filenames = set()

    for split in ["train", "val"]:

        folder = os.path.join(
            SPLIT_DIR,
            split,
            class_name
        )

        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Could not find: {folder}"
            )

        for filename in os.listdir(folder):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                filenames.add(
                    os.path.basename(filename).strip()
                )

    return filenames


A_names = get_filenames("A")
B_names = get_filenames("B")


print("A in split_data:", len(A_names))
print("B in split_data:", len(B_names))


# ============================================================
# MARK A / B
# ============================================================

A = df[
    df["filename"].isin(A_names)
]

B = df[
    df["filename"].isin(B_names)
]


print("A found in PCA:", len(A))
print("B found in PCA:", len(B))


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(10, 10)
)


# All ResNet PCA points
plt.scatter(
    df["PC1"],
    df["PC2"],
    c="lightgray",
    s=5,
    alpha=0.25,
    label="All images"
)


# Group A
plt.scatter(
    A["PC1"],
    A["PC2"],
    c="red",
    s=15,
    alpha=0.8,
    label="A"
)


# Group B
plt.scatter(
    B["PC1"],
    B["PC2"],
    c="blue",
    s=15,
    alpha=0.8,
    label="B"
)


plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "FaceNet-defined A/B groups in ResNet50 PCA space"
)

plt.legend()

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()


print("Saved:", OUTPUT_PATH)