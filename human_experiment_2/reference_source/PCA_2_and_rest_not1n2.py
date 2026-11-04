
####################################################################################################
# PCA Analysis of FaceNet Embeddings
#
# This script performs PCA on 512D face embeddings extracted by FaceNet.
#
# It allows selecting any two principal components to define a 2D stimulus space.
# The remaining components (within the 95% explained-variance range) are used
# to filter images according to their squared distance from the residual origin.
#
# Outputs:
# - pca_PC{A}_PC{B}_filtered.csv
#     Selected images projected onto the chosen 2D PCA plane.
#
# - pca_eigenvalues.csv (optional)
#     Eigenvalues and explained variance of all components.
#
# The filtered CSV has no header and contains:
# filename, PC_A coordinate, PC_B coordinate
####################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ================================================================================================
# CONFIGURATION
# ================================================================================================

INPUT_PATH = "female_facenet_vggface2_embeddings.csv"

# Select the two principal components (1-based indexing)
PC_A = 3
PC_B = 4

# Percentage of images retained after residual-space filtering
FILTER_PERCENT = 10

# Output filename
OUTPUT_PATH = f"pca_PC{PC_A}_PC{PC_B}_filtered.csv"


# ================================================================================================
# LOAD DATA
# ================================================================================================

def load_data(path):
    data = []
    names = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 2:
                continue

            name = parts[0]
            vec = list(map(float, parts[1:]))

            names.append(name)
            data.append(vec)

    return np.array(data), names


X, names = load_data(INPUT_PATH)

print("Shape of embeddings:", X.shape)
print("Number of images:", len(names))


# ================================================================================================
# CENTER DATA
# ================================================================================================

def center_data(X):
    mean = np.mean(X, axis=0)
    return X - mean


X_centered = center_data(X)

# Use centered embeddings without standardization,
# preserving the original PCA methodology.
X_prepared = X_centered


# ================================================================================================
# COMPUTE PCA
# ================================================================================================

def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)


covariance_matrix = compute_covariance_matrix(X_prepared)

print("Covariance matrix shape:", covariance_matrix.shape)


def compute_eigenvalues_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(
    covariance_matrix
)


def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors


sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues_eigenvectors(
    eigenvalues,
    eigenvectors
)


# ================================================================================================
# EXPLAINED VARIANCE
# ================================================================================================

total_variance = np.sum(sorted_eigenvalues)

explained_variance = sorted_eigenvalues / total_variance * 100
cumulative_variance = np.cumsum(explained_variance)

# Number of components required to explain at least 95% of the variance
k_95 = np.argmax(cumulative_variance >= 95) + 1

print(f"Components required for 95% variance: {k_95}")

print(
    f"Selected components: PC{PC_A}, PC{PC_B}"
)

print(
    f"Explained variance of PC{PC_A}: "
    f"{explained_variance[PC_A - 1]:.2f}%"
)

print(
    f"Explained variance of PC{PC_B}: "
    f"{explained_variance[PC_B - 1]:.2f}%"
)


# ================================================================================================
# PLOT EXPLAINED VARIANCE
# ================================================================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="."
)

plt.axhline(95, color="red", linestyle="--", label="95% variance")

plt.title("Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance (%)")

plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(sorted_eigenvalues) + 1),
    sorted_eigenvalues,
    marker=".",
    label="Eigenvalues"
)

plt.title("Eigenvalues Distribution")
plt.xlabel("Principal Components")
plt.ylabel("Eigenvalue")

plt.grid(True)
plt.legend()
plt.show()


# ================================================================================================
# PROJECT DATA ONTO PCA COMPONENTS
# ================================================================================================

def project_data(X, eigenvectors, component_indices):
    return X @ eigenvectors[:, component_indices]


# Validate selected components
if PC_A == PC_B:
    raise ValueError("PC_A and PC_B must be different.")

if not (1 <= PC_A <= k_95 and 1 <= PC_B <= k_95):
    raise ValueError(
        f"Selected components must be between 1 and {k_95}."
    )


# Convert component numbers to zero-based indices
selected_indices = [PC_A - 1, PC_B - 1]

# Project onto the selected 2D plane
X_projected_2d = project_data(
    X_prepared,
    sorted_eigenvectors,
    selected_indices
)

print("2D projection shape:", X_projected_2d.shape)


# ================================================================================================
# RESIDUAL PCA SPACE
# ================================================================================================

# Use all components within the 95% variance range,
# excluding the two components selected for the 2D plane.

remaining_indices = [
    i for i in range(k_95)
    if i not in selected_indices
]

X_projected_rest = project_data(
    X_prepared,
    sorted_eigenvectors,
    remaining_indices
)

print("Residual projection shape:", X_projected_rest.shape)


# ================================================================================================
# FILTER IMAGES BY RESIDUAL DISTANCE
# ================================================================================================

# Squared Euclidean distance from the origin in residual PCA space
squared_radius_rest = np.sum(
    X_projected_rest ** 2,
    axis=1
)

# Keep the FILTER_PERCENT images with the smallest residual distances
threshold_sq = np.percentile(
    squared_radius_rest,
    FILTER_PERCENT
)

mask_sq = squared_radius_rest <= threshold_sq

print(
    f"Images retained after filtering: "
    f"{np.sum(mask_sq)} / {len(names)}"
)


# ================================================================================================
# CREATE DATAFRAMES
# ================================================================================================

# Full 2D projection
df_2d = pd.DataFrame(
    X_projected_2d,
    columns=[f"PC{PC_A}", f"PC{PC_B}"]
)

df_2d.insert(0, "filename", names)


# Residual projection
rest_cols = [
    f"PC{i + 1}" for i in remaining_indices
]

df_rest = pd.DataFrame(
    X_projected_rest,
    columns=rest_cols
)

df_rest.insert(0, "filename", names)


# Filtered 2D projection
df_2d_filtered = df_2d.loc[mask_sq].copy()


# ================================================================================================
# SAVE RESULTS
# ================================================================================================

# Save filtered 2D coordinates
# No header, matching the format of the original experiment.
df_2d_filtered.to_csv(
    OUTPUT_PATH,
    index=False,
    header=False
)

print(f"Saved filtered 2D projection to: {OUTPUT_PATH}")


# Optional: save PCA eigenvalues and explained variance
eigenvalues_df = pd.DataFrame(
    {
        "component": np.arange(1, len(sorted_eigenvalues) + 1),
        "eigenvalue": sorted_eigenvalues,
        "explained_variance_percent": explained_variance,
        "cumulative_variance_percent": cumulative_variance,
    }
)

# eigenvalues_df.to_csv("pca_eigenvalues.csv", index=False)


# ================================================================================================
# VISUALIZE SELECTED PCA PLANE
# ================================================================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    X_projected_2d[:, 0],
    X_projected_2d[:, 1],
    s=2,
    alpha=0.15,
    label="All images"
)

plt.scatter(
    X_projected_2d[mask_sq, 0],
    X_projected_2d[mask_sq, 1],
    s=4,
    alpha=0.6,
    label=f"Filtered {FILTER_PERCENT}%"
)

plt.xlabel(f"PC{PC_A}")
plt.ylabel(f"PC{PC_B}")

plt.title(
    f"Face Embeddings: PC{PC_A} vs PC{PC_B}"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
