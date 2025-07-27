#################################################################
# Compute and plot the mean and standard deviation of each feature
# across FaceNet embedding vectors (each of length 512)
# 12/05/2025
#################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file (first column = image names, remaining = embedding vectors)
df = pd.read_csv("24_000/celeba_facenet_embeddings_997.csv", header=None)

# Separate names and vectors
names = df.iloc[:, 0]
vectors = df.iloc[:, 1:].to_numpy()  # shape: (997, 512)

# Compute per-feature mean and standard deviation
mean_vector = np.mean(vectors, axis=0)  # shape: (512,)
std_vector = np.std(vectors, axis=0)  # shape: (512,)

# Save results to CSV
summary_df = pd.DataFrame({"mean": mean_vector, "std": std_vector})
summary_df.to_csv("embedding_statistics.csv", index_label="feature_index")
print("Saved per-feature mean and std to embedding_statistics.csv")

# Plot histogram of means
plt.figure(figsize=(10, 4), num="mean_distribution")
plt.hist(mean_vector, bins=50, edgecolor="black")
plt.title("Distribution of Means of Vector Entries")
plt.xlabel("Mean Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Plot histogram of standard deviations
plt.figure(figsize=(10, 4), num="std_distribution")
plt.hist(std_vector, bins=50, edgecolor="black")
plt.title("Distribution of Standard Deviations of Vector Entries")
plt.xlabel("Standard Deviation")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
