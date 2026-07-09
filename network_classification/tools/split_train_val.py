############################################################
# this script splits the dataset into training and validation sets
# It assumes the dataset is organized in two main folders: "A" and "B"
# The script will create a new directory structure with "train" and "val" folders
############################################################
import os
import shutil
import random
from tqdm import tqdm


def split_dataset(source_dir, target_dir, split_ratio=0.8):
    """
    Splits the dataset into training and validation sets.

    Args:
        source_dir (str): The source directory containing sets A and B.
        target_dir (str): The target directory to save the split datasets.
        split_ratio (float): The ratio of training data to validation data (default is 0.8).
    """
    classes = os.listdir(source_dir)
    print(f"Classes found: {classes}")
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        print(f"Processing class: {cls} from {class_dir}")
        images = os.listdir(class_dir)
        print(f"Number of images in {cls}: {len(images)}")
        random.shuffle(images)
        split_index = int(len(images) * split_ratio)
        print(
            f"Split index for {cls}: {split_index} (train: {len(images[:split_index])}, val: {len(images[split_index:])})"
        )
        train_images = images[:split_index]
        val_images = images[split_index:]

        train_dir = os.path.join(target_dir, "train", cls)
        os.makedirs(train_dir, exist_ok=True)
        print(f"Copying training images to: {train_dir}")
        for img in tqdm(train_images, desc=f"Train/{cls}", leave=False):
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_dir, img)
            shutil.copy(src, dst)
        print(f"Copied {len(train_images)} images to {train_dir}")

        val_dir = os.path.join(target_dir, "val", cls)
        os.makedirs(val_dir, exist_ok=True)
        print(f"Copying validation images to: {val_dir}")
        for img in tqdm(val_images, desc=f"Val/{cls}", leave=False):
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.copy(src, dst)
        print(f"Copied {len(val_images)} images to {val_dir}")

    print("\nDataset splitting completed.")


split_dataset("original_data", "split_data", split_ratio=0.8)

print("Train A:", len(os.listdir("split_data/train/A")))
print("Val A:", len(os.listdir("split_data/val/A")))
print("Train B:", len(os.listdir("split_data/train/B")))
print("Val B:", len(os.listdir("split_data/val/B")))
