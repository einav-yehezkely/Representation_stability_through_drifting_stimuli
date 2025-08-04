import pandas as pd
import random

df_a = pd.read_csv("rotation_sequence_A.csv")
df_b = pd.read_csv("rotation_sequence_B.csv")

i, j = 0, 0
rows = []

while i < len(df_a) or j < len(df_b):
    if i == len(df_a):
        row = df_b.iloc[j].copy()
        row["group"] = "B"
        rows.append(row)
        j += 1
    elif j == len(df_b):
        row = df_a.iloc[i].copy()
        row["group"] = "A"
        rows.append(row)
        i += 1
    else:
        if random.random() < 0.5:
            row = df_a.iloc[i].copy()
            row["group"] = "A"
            rows.append(row)
            i += 1
        else:
            row = df_b.iloc[j].copy()
            row["group"] = "B"
            rows.append(row)
            j += 1

# Convert list of Series to DataFrame
merged_df = pd.DataFrame(rows)

# Move 'group' column to the front
cols = ["group"] + [col for col in merged_df.columns if col != "group"]
merged_df = merged_df[cols]

# Save to CSV
merged_df.to_csv("merged_sequences.csv", index=False)
