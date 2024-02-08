import os
import sys
import subprocess
import json

import random


def process_directory(inerf_transforms_path, original_transform_path, unreg_transform_path, inerf=True, seed=42):
    with open(os.path.join(original_transform_path), 'r') as f:
        original_transform = json.load(f)
    with open(os.path.join(unreg_transform_path), 'r') as f:
        unreg_transform = json.load(f)
    original_frames = original_transform["frames"]

    # for frame in original_frames:
    #     frame["file_path"] = "../" + dir_name + "/" + frame["file_path"]

    # Sample single frame
    if inerf:
        random.seed(seed)
        original_frames = [original_frames[random.randint(0, len(original_frames)-1)]]

    inerf_transform = original_transform.copy()
    inerf_transform["frames"] = original_frames
    inerf_transform["transform"] = unreg_transform["transform"]
    inerf_transform["scale"] = unreg_transform["scale"]
    inerf_transform["registration_matrix"] = unreg_transform["registration_matrix"]
    inerf_transform["registration_rot_euler"] = unreg_transform["registration_rot_euler"]
    inerf_transform["registration_translation"] = unreg_transform["registration_translation"]
    print(f"[INFO] writing {len(inerf_transform['frames'])} frames to {inerf_transforms_path}")
    with open(inerf_transforms_path, "w") as outfile:
        json.dump(inerf_transform, outfile, indent=2)



if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_data.py <inerf_transforms_path> <original_transform_path> <unreg_transform_path> <seed>")
    else:
        inerf_transforms_path = sys.argv[1]
        original_transform_path = sys.argv[2]
        unreg_transform_path = sys.argv[3]
        seed = sys.argv[4]
        process_directory(inerf_transforms_path, original_transform_path, unreg_transform_path, int(seed))
