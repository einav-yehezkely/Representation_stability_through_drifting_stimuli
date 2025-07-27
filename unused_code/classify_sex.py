#################################################################
# This script makes different folders for men and women and moves the images from celebA to the respective folders.
#################################################################
import os
import shutil
import pandas as pd
from tqdm import tqdm  # progress bar

# === Configuration ===
images_dir = "celebA"
attr_file = "list_attr_celeba.txt"
male_dir = "male_faces"
female_dir = "female_faces"

# === Load attribute data ===
df = pd.read_csv(attr_file, sep="\s+", skiprows=1)
df.index.name = "image_name"

# === Create output folders if needed ===
os.makedirs(male_dir, exist_ok=True)
os.makedirs(female_dir, exist_ok=True)

# === Copy images with a progress bar ===
for image_name, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
    gender = row["Male"]
    src_path = os.path.join(images_dir, image_name)

    if not os.path.exists(src_path):
        continue  # skip if image not found

    dst_dir = male_dir if gender == 1 else female_dir
    dst_path = os.path.join(dst_dir, image_name)

    shutil.copyfile(src_path, dst_path)

print("Done!")
