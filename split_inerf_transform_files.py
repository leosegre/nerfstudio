import os
import sys
import subprocess
import json

import random


def process_directory(dir_name, output_directory):
    exp_types = ["0_100_even_odd", "30_70_even_odd", "50_50"]
    transforms_path = os.path.join(output_directory, dir_name, 'transforms.json')
    with open(os.path.join(transforms_path), 'r') as f:
        transforms = json.load(f)
    original_frames = transforms["frames"]

    # for frame in original_frames:
    #     frame["file_path"] = "../" + dir_name + "/" + frame["file_path"]

    # Sample single frame
    original_frames = [original_frames[random.randint(0, len(original_frames)-1)]]

    for exp in exp_types:
        unreg_transforms_path = os.path.join(output_directory, dir_name+"_"+exp+"_unregistered", 'transforms.json')
        with open(os.path.join(unreg_transforms_path), 'r') as f:
            unreg_transforms = json.load(f)
        inerf_transforms = unreg_transforms.copy()
        inerf_transforms["frames"] = original_frames
        inerf_transforms_path = os.path.join(output_directory, dir_name, 'inerf_' + exp + '_transforms.json')
        print(f"[INFO] writing {len(inerf_transforms['frames'])} frames to {inerf_transforms_path}")
        with open(inerf_transforms_path, "w") as outfile:
            json.dump(inerf_transforms, outfile, indent=2)


def main(base_directory, output_directory):
    for dir_name in os.listdir(base_directory):
        if dir_name == "leaves":  ## FIXME
            continue
        full_path = os.path.join(base_directory, dir_name)
        if os.path.isdir(full_path):
            process_directory(dir_name, output_directory)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <base_directory> <output_directory>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            main(base_directory, output_directory)
