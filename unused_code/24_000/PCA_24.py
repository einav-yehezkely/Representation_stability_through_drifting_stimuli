#################################################################
# Representation of the face in K_95 dimensions using PCA on 512 dimensions vectors
# 11/05/2025
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "celeba_facenet_embeddings_997.csv"  # Path to the data file


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
def project_data(X, eigenvectors, k):
    return X @ eigenvectors[:, :k]


# Project the centered data onto the top k_95 (=58) principal components
# This reduces the dimensionality of the original 128D vectors to 58D while preserving ~95% of the variance
X_projected = project_data(X_prepared, sorted_eigenvectors, k_95)
print("shape (X_projected):", X_projected.shape)

# Create a DataFrame with the projected data and include the original image filenames
df_projected = pd.DataFrame(X_projected)
df_projected.insert(0, "filename", names)

# Save the projected data to a CSV file
df_projected.to_csv("pca_95_percent_variance.csv", index=False)
print("Saved PCA-reduced embeddings to pca_95_faces.csv")

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
