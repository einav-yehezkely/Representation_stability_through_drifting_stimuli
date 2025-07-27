#################################################################
# 2D_vectors_scatter.py
# This script reads a CSV file containing 2D PCA vectors and creates a scatter plot of the vectors.
# It uses the first column as names and the second and third columns as the x and y coordinates, respectively.
#################################################################

import pandas as pd
import matplotlib

# matplotlib.use("Agg")  # Use a non-interactive backend (no GUI)
import matplotlib.pyplot as plt

df = pd.read_csv("pca_top2_filtered_female.csv", header=None)

print(f"Number of vectors: {df.shape[0]}")
print(f"Vector dimensionality: {df.shape[1] - 1}")  # Subtract 1 for the names column

# first column = names, second column = dimension 1, third column = dimension 2
x = df.iloc[:, 1]
y = df.iloc[:, 2]

plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=1, alpha=0.3)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.title("2D Image Vectors")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
# plt.savefig("scatter_plot.png", dpi=300)
# print("Plot saved to scatter_plot.png")
# plt.savefig("scatter_plot.svg")
# print("SVG plot saved – open it in a browser or vector image viewer.")
