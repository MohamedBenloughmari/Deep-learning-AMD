import os
import shutil
import math
import random

random.seed(42)  # for reproducible sampling

SOURCE_ROOT = "data_out/Task_2"
TARGET_ROOT = "data_out10/Task2"
SPLITS = ["train", "test", "val"]
FRACTION = 0.10  # 10%

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sample_and_copy_split(split_name: str):
    src_split_dir = os.path.join(SOURCE_ROOT, split_name)
    dst_split_dir = os.path.join(TARGET_ROOT, split_name)

    # Walk through all subdirectories (classes) in this split
    for root, dirs, files in os.walk(src_split_dir):
        # Compute corresponding target directory
        rel_path = os.path.relpath(root, src_split_dir)
        target_dir = os.path.join(dst_split_dir, rel_path)
        ensure_dir(target_dir)

        if not files:
            continue

        # Sample 10% of the files (at least 1 if there are files)
        n_files = len(files)
        n_sample = max(1, math.floor(n_files * FRACTION))
        sampled_files = random.sample(files, n_sample)

        for fname in sampled_files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_dir, fname)
            shutil.copy2(src_path, dst_path)

def main():
    for split in SPLITS:
        sample_and_copy_split(split)
    print("Done creating 10% dataset at:", TARGET_ROOT)

if __name__ == "__main__":
    main()