import os
import glob
import random
import shutil

original_root = "data"
bag_root = "data_bags"
bag_size = 16

os.makedirs(bag_root, exist_ok=True)

for label in ["ALL", "NORMAL"]:
    img_paths = glob.glob(os.path.join(original_root, label, "*.png"))
    random.shuffle(img_paths)
    num_bags = len(img_paths) // bag_size
    os.makedirs(os.path.join(bag_root, label), exist_ok=True)

    for i in range(num_bags):
        bag_folder = os.path.join(bag_root, label, f"bag_{i+1}")
        os.makedirs(bag_folder, exist_ok=True)
        for j in range(bag_size):
            src = img_paths[i*bag_size + j]
            dst = os.path.join(bag_folder, os.path.basename(src))
            shutil.copy(src, dst)
