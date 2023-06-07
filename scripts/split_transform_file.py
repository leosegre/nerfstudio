import sys
import shutil
import os
import json
import numpy as np

def main(argv):
    dir_name = argv
    transforms_path = os.path.join(dir_name, 'transforms.json')
    transforms1_path = os.path.join(dir_name, 'transforms1.json')
    transforms2_path = os.path.join(dir_name, 'transforms2.json')
    backup_file_path = os.path.join(dir_name, 'transforms_backup')
    if not os.path.exists(backup_file_path):
        os.mkdir(backup_file_path)
    shutil.copy(transforms_path, backup_file_path)

    with open(os.path.join(transforms_path), 'r') as f:
        transforms = json.load(f)

    transforms1 = transforms.copy()
    transforms2 = transforms.copy()
    frames1 = []
    frames2 = []

    for i, frame in enumerate(transforms["frames"]):
        frames_len = len(transforms["frames"])
        # if frame["colmap_im_id"] <= frames_len * 0.7:
        if frame["colmap_im_id"] % 2:
            frames1.append(frame)
        # if frame["colmap_im_id"] > frames_len * 0.3:
        if frame["colmap_im_id"] % 2 == 0:
            frames2.append(frame)

    transforms1["frames"] = frames1
    transforms2["frames"] = frames2

    print(f"[INFO] writing {len(transforms1['frames'])} frames to {transforms1_path}")
    with open(transforms1_path, "w") as outfile:
        json.dump(transforms1, outfile, indent=2)

    print(f"[INFO] writing {len(transforms2['frames'])} frames to {transforms2_path}")
    with open(transforms2_path, "w") as outfile:
        json.dump(transforms2, outfile, indent=2)

if __name__ == "__main__":
   main(sys.argv[1])