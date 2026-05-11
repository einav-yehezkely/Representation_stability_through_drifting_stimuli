import pandas as pd

# load log
df = pd.read_csv("training_log.csv")

# sort for correct temporal comparison
df = df.sort_values(["filename", "iteration"]).reset_index(drop=True)

# binary decision from prob_A
df["pred"] = (df["prob_A"] >= 0.5).astype(int)  # 1 = A, 0 = B

# previous iteration info for the same image
df["prev_iteration"] = df.groupby("filename")["iteration"].shift(1)
df["prev_prob_A"] = df.groupby("filename")["prob_A"].shift(1)
df["prev_pred"] = df.groupby("filename")["pred"].shift(1)

# keep only consecutive iterations: t -> t+1
df["is_consecutive"] = df["iteration"] == df["prev_iteration"] + 1

# keep only true label changes
changed = df[
    df["prev_iteration"].notna()
    & df["is_consecutive"]
    & (df["pred"] != df["prev_pred"])
].copy()

# nicer columns
changed["from_label"] = changed["prev_pred"].map({1: "A", 0: "B"})
changed["to_label"] = changed["pred"].map({1: "A", 0: "B"})
changed["from_iteration"] = changed["prev_iteration"].astype(int)
changed["to_iteration"] = changed["iteration"].astype(int)

# final table
changed_csv = changed[
    [
        "filename",
        "from_iteration",
        "to_iteration",
        "from_label",
        "to_label",
        "prev_prob_A",
        "prob_A",
    ]
].rename(
    columns={
        "prev_prob_A": "from_prob_A",
        "prob_A": "to_prob_A",
    }
)

# optional: add probability delta
changed_csv["delta_prob_A"] = changed_csv["to_prob_A"] - changed_csv["from_prob_A"]

# save all changes
changed_csv.to_csv("changed_images_between_consecutive_iterations.csv", index=False)

# also save summary: how many changed in each t+1
summary = (
    changed_csv.groupby("to_iteration")["filename"]
    .count()
    .reset_index(name="num_changed_images")
)
summary.to_csv("changed_images_summary_per_iteration.csv", index=False)

print("Saved:")
print("1. changed_images_between_consecutive_iterations.csv")
print("2. changed_images_summary_per_iteration.csv")

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import numpy as np

# =========================
# Number of changed images per iteration
# with linear regression
# =========================

x = summary["to_iteration"].values.reshape(-1, 1)
y = summary["num_changed_images"].values

# fit linear regression
model = LinearRegression()
model.fit(x, y)

# predicted line
y_pred = model.predict(x)

plt.figure(figsize=(10, 5))

# scatter only (no connecting lines)
plt.scatter(x, y, s=15)

# regression line
plt.plot(
    x,
    y_pred,
    color="red",
    linestyle="--",
    label=f"Linear Fit (slope={model.coef_[0]:.2f})",
)

plt.xlabel("Iteration")
plt.ylabel("Number of images that changed label")
plt.title("Label changes between consecutive iterations")
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.savefig("changed_images_summary_per_iteration_with_regression.png", dpi=300)
plt.show()
