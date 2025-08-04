############################################################
# This script checks for identical filenames in two directories
# it is used to ensure that there is no leakage of data
# between training and validation datasets.
############################################################
import os

# Paths to the two directories
dir1 = "split_data/train/A"
dir2 = "split_data/val/A"

# Get sets of filenames from both directories
files1 = set(os.listdir(dir1))
files2 = set(os.listdir(dir2))

# Find common filenames
common_files = files1.intersection(files2)

# Print results
if common_files:
    print("Found identical filenames in both folders:")
    for filename in sorted(common_files):
        print(filename)
else:
    print("No identical filenames found.")
