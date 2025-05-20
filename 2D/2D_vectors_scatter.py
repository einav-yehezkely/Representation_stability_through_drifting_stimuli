#################################################################
# 2D_vectors_scatter.py
# This script reads a CSV file containing 2D PCA vectors and creates a scatter plot of the vectors.
# It uses the first column as names and the second and third columns as the x and y coordinates, respectively.
#################################################################

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pca_2D_24.csv", header=None)

# first column = names, second column = dimension 1, third column = dimension 2
x = df.iloc[:, 1]
y = df.iloc[:, 2]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=10, alpha=0.7)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.title("2D Image Vectors")
plt.grid(True)
plt.axis("equal")
plt.show()
