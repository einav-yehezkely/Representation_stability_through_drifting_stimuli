import pandas as pd
import os

df = pd.read_csv("female_arcface_embeddings.csv", header=None)

csv_names = set(
    df.iloc[:, 0]
    .astype(str)
    .apply(os.path.basename)
    .str.strip()
)

print("201163.jpg in embeddings:", "201163.jpg" in csv_names)
print("201163.jpg exists in female_faces:", os.path.exists("female_faces/201163.jpg"))

image_names = {
    f for f in os.listdir("female_faces")
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

missing = sorted(image_names - csv_names)

print("Images in female_faces:", len(image_names))
print("Unique filenames in embeddings:", len(csv_names))
print("Missing embeddings:", len(missing))
print("First missing images:", missing[:20])