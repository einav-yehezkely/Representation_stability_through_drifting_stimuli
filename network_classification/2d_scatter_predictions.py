#################################################################
# 2D_vectors_colored_scatter.py
# This script reads a CSV file containing all 2D PCA vectors and displays them in gray,
# then colors the ones predicted as A (blue) and B (red) based on image names.
#################################################################

import pandas as pd
import matplotlib.pyplot as plt

# Load full embeddings: name, x, y
df = pd.read_csv("pca_top2_filtered_female.csv", header=None)
df.columns = ["name", "x", "y"]

# Load predicted names (from column 4 of each file)
pred_a = pd.read_csv("predicted_as_A.csv", header=None)[3]
pred_b = pd.read_csv("predicted_as_B.csv", header=None)[3]

cluster_a = pd.read_csv("filenames_A.csv", header=None)[0]
cluster_b = pd.read_csv("filenames_B.csv", header=None)[0]

# Filter by predictions
df_a = df[df["name"].isin(pred_a)]
df_b = df[df["name"].isin(pred_b)]

df_cluster_a = df[df["name"].isin(cluster_a)]
df_cluster_b = df[df["name"].isin(cluster_b)]

print(f"Total vectors: {len(df)}")
print(f"Predicted A: {len(df_a)}")
print(f"Predicted B: {len(df_b)}")
print(f"Cluster A: {len(df_cluster_a)}")
print(f"Cluster B: {len(df_cluster_b)}")

# Plot
plt.figure(figsize=(10, 10))

# All points in gray
plt.scatter(df["x"], df["y"], s=5, alpha=0.3, color="gray", label="All Vectors")

plt.scatter(
    df_cluster_a["x"],
    df_cluster_a["y"],
    s=9,
    alpha=0.8,
    color="lightblue",
    label="Trained A",
)
# Predicted A in blue
plt.scatter(df_a["x"], df_a["y"], s=10, alpha=0.7, color="blue", label="Predicted A")

plt.scatter(
    df_cluster_b["x"],
    df_cluster_b["y"],
    s=9,
    alpha=0.7,
    color="pink",
    label="Trained B",
)
# Predicted B in red
plt.scatter(df_b["x"], df_b["y"], s=10, alpha=0.7, color="red", label="Predicted B")

# Axes and labels
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.title("Model Predictions")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig("45_225 predictions.png", dpi=300)
plt.show()
