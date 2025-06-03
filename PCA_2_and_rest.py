#################################################################
#  This script projects face embeddings (512D) onto PCA components 3 to k_95
# 28/05/2025
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "celeba_facenet_embeddings_24.csv"  # Path to the data file


# Load the data
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


X, names = load_data(path)
print("shape (X):", X.shape)
print("num of pics", len(names))


# Center the data
def center_data(X):
    mean = np.mean(X, axis=0)
    return X - mean


X_centered = center_data(X)


# normalize the data
def normalize_data(X):
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return X / std


X_prepared = X_centered


# compute covariance matrix
def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)


covariance_matrix = compute_covariance_matrix(X_prepared)
print("shape (covariance_matrix):", covariance_matrix.shape)


# compute eigenvalues and eigenvectors
def compute_eigenvalues_eigenvectors(correlation_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(covariance_matrix)


# sort eigenvalues and eigenvectors
def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors


sorted_eigenvalues, sorted_eigenvectors = sort_eigenvalues_eigenvectors(
    eigenvalues, eigenvectors
)

# Compute explained variance (percentage)
total_variance = np.sum(sorted_eigenvalues)
explained_variance = sorted_eigenvalues / total_variance * 100
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker=".")
plt.title("Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance (%)")
plt.grid(True)
plt.show()

# distribution of eigenvalues
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(sorted_eigenvalues) + 1),
    sorted_eigenvalues,
    marker=".",
    label="Eigenvalues",
)
plt.title("Eigenvalues Distribution")
plt.xlabel("Principal Components")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.legend()
plt.show()

k_95 = np.argmax(cumulative_variance >= 95) + 1
print(f"need {k_95} components to explain at least 95% of the variance.")


# project the data onto the k_95 eigenvectors that match the largest eigenvalues
def project_data(X, eigenvectors, start=0, end=None):
    if end is None:
        end = eigenvectors.shape[1]
    return X @ eigenvectors[:, start:end]


# Project the centered data onto the top k_95 principal components
# This reduces the dimensionality of the original 512D vectors to k_95D while preserving ~95% of the variance
X_projected = project_data(X_prepared, sorted_eigenvectors, k_95)
print("shape (X_projected):", X_projected.shape)
# Project the centered data onto the top 2 principal components
X_projected_top2 = project_data(X_prepared, sorted_eigenvectors, 0, 2)
# project the rest of the principal components
X_projected_rest = project_data(X_prepared, sorted_eigenvectors, 2, k_95)


# Create a DataFrame with the projected data and include the original image filenames
df_projected = pd.DataFrame(X_projected)
df_projected.insert(0, "filename", names)
df_top2 = pd.DataFrame(X_projected_top2, columns=["PC1", "PC2"])
df_top2.insert(0, "filename", names)
rest_cols = [f"PC{i}" for i in range(3, k_95 + 1)]
df_rest = pd.DataFrame(X_projected_rest, columns=rest_cols)
df_rest.insert(0, "filename", names)

# Save the projected data to a CSV file
df_projected.to_csv("pca_95_percent_variance.csv", index=False)
print("Saved PCA-reduced embeddings to pca_95_faces.csv")
df_top2.to_csv("pca_top2.csv", index=False)
print("Saved projection onto top 2 components to pca_top2.csv")
df_rest.to_csv("pca_rest.csv", index=False)
print("Saved projection onto components 3 to k_95 to pca_rest.csv")

# Save eigenvalues to CSV
eigenvalues_df = pd.DataFrame(
    {
        "component": np.arange(1, len(sorted_eigenvalues) + 1),
        "eigenvalue": sorted_eigenvalues,
        "explained_variance_percent": explained_variance,
        "cumulative_variance_percent": cumulative_variance,
    }
)
eigenvalues_df.to_csv("pca_eigenvalues.csv", index=False)
print("Saved eigenvalues to pca_eigenvalues.csv")

# compute squared radius for the rest of the components
squared_radius_rest = np.sum(X_projected_rest**2, axis=1)
# Select 5% of vectors with smallest squared radius — most similar in non-dominant components
threshold_sq = np.percentile(squared_radius_rest, 5)
mask_sq = squared_radius_rest <= threshold_sq
# Apply the mask to get only the 5% most similar vectors in the residual space
similar_vectors_rest = X_projected_rest[mask_sq]

rest_cols = [f"PC{i}" for i in range(3, k_95 + 1)]
df_similar_vectors = pd.DataFrame(similar_vectors_rest, columns=rest_cols)
df_similar_vectors.insert(0, "filename", np.array(names)[mask_sq])
df_similar_vectors.to_csv("pca_rest_vectors_5_percent.csv", index=False)
print(
    "Saved 5% of images with smallest squared radius in residual PCA space to pca_rest_vectors_5_percent.csv"
)

df_top2_filtered = df_top2[mask_sq]
df_top2_filtered.to_csv("pca_top2_filtered.csv", index=False, header=False)
print(
    "Saved filtered projection (PC1, PC2) of 5% most similar images to pca_top2_filtered.csv"
)
