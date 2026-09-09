import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "pca_top2_filtered_female_vgg_1.csv",
    header=None
)

print(f"Number of vectors: {df.shape[0]}")
print(f"Vector dimensionality: {df.shape[1] - 1}")

# first column = filename
# second column = PC1
# third column = PC2
x = df.iloc[:, 1]
y = df.iloc[:, 2]

plt.figure(figsize=(10, 10))

plt.scatter(
    x,
    y,
    s=1,
    color="gray",
    alpha=0.3
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)

plt.title(
    "2D PCA Projection of Female Face Embeddings\n"
    "(Filtered for Low Residual Variance)"
)

plt.grid(True)
plt.axis("equal")
plt.tight_layout()

# Save BEFORE show
plt.savefig(
    "scatter_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "scatter_plot.svg",
    bbox_inches="tight"
)

print("Plots saved.")

plt.show()