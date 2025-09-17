#################################################################
# 2D_vectors_colored_scatter.py
# This script reads a CSV file containing all 2D PCA vectors and displays them in gray,
# then colors the ones predicted as A (blue) and B (red) based on image names.
#################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

angle = 135
opposite_angle = (angle + 180) % 360

# Load full embeddings: name, x, y
df = pd.read_csv("pca_top2_filtered_female.csv", header=None)
df.columns = ["name", "x", "y"]

# Load predicted names (from column 4 of each file)
pred_a = pd.read_csv("predicted_as_A.csv")["filename"]
pred_b = pd.read_csv("predicted_as_B.csv")["filename"]


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


# Add circle and lines for reference
radius = max(np.sqrt(df["x"] ** 2 + df["y"] ** 2)) * 1.05
circle = Circle((0, 0), radius, fill=False, color="black", linestyle="--", alpha=0.5)
plt.gca().add_patch(circle)
for angle_circ in range(0, 360, 20):
    rad = np.deg2rad(angle_circ)
    x = radius * np.cos(rad)
    y = radius * np.sin(rad)
    plt.plot([0, x], [0, y], color="gray", linewidth=0.5, alpha=0.5)
    plt.text(x * 1.05, y * 1.05, f"{angle_circ}°", ha="center", va="center")

# Axes and labels
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.title(
    f"images predicted as A/B, trained on {angle}° and {opposite_angle}° clusters"
)
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.savefig(f"{angle}_{opposite_angle} predictions.png", dpi=300)
plt.show()


# linear graph
pred_a = pd.read_csv("predicted_as_A.csv")
pred_b = pd.read_csv("predicted_as_B.csv")

pred_a["pred"] = "A"
pred_b["pred"] = "B"

df = pd.concat([pred_a, pred_b], ignore_index=True)

df_full = pd.read_csv("pca_top2_filtered_female.csv", header=None)
df_full.columns = ["filename", "x", "y"]

df = df.merge(df_full, on="filename", how="left")

window_size = 20
results = []

for step_angle in range(0, 360, 1):
    end = (step_angle + window_size) % 360
    if step_angle < end:
        window_data = df[(df["angle_deg"] >= step_angle) & (df["angle_deg"] < end)]
    else:
        window_data = df[(df["angle_deg"] >= step_angle) | (df["angle_deg"] < end)]

    total = len(window_data)

    if total > 0:
        count_a = (window_data["pred"] == "A").sum()
        count_b = (window_data["pred"] == "B").sum()
        percent_a = count_a * 100 / total
        percent_b = count_b * 100 / total
    else:
        percent_a = 0
        percent_b = 0

    # Use center of window for plotting
    center_angle = (step_angle + window_size / 2) % 360
    results.append((center_angle, percent_a, percent_b))

df_results = pd.DataFrame(results, columns=["angle", "percent_A", "percent_B"])

# Add closing point at 360°
angle0 = df_results[df_results["angle"] == 0]
# 360° is the same as 0°
angle360 = angle0.copy()
angle360["angle"] = 360
df_results = pd.concat([df_results, angle360], ignore_index=True)
# sort by angle for proper plotting
df_results = df_results.sort_values(by="angle")

plt.figure(figsize=(12, 6))
plt.plot(
    df_results["angle"], df_results["percent_A"], label="Predicted A", color="blue"
)
plt.plot(df_results["angle"], df_results["percent_B"], label="Predicted B", color="red")
plt.xlabel("Angle")
plt.ylabel("%")
plt.title(
    f"% images predicted as A/B, trained on {angle}° and {opposite_angle}° clusters, {window_size}° slices"
)
plt.axhline(y=0, color="black", linewidth=1)
plt.axvline(x=0, color="black", linewidth=1)
plt.axvline(x=angle, color="blue", linewidth=1, linestyle="--")
plt.axvline(x=opposite_angle, color="red", linewidth=1, linestyle="--")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.ylim(0, 100)
plt.xlim(0, 360)
plt.savefig(f"{angle}_{opposite_angle} predictions_linear.png", dpi=300)
plt.show()
