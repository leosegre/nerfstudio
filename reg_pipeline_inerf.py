import os
import subprocess
from datetime import datetime
import sys
import numpy as np

import json


def main(data_dir, outputs_dir, scene_names, exp_types, downscale, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        reconstruct_scenes = True
    else:
        reconstruct_scenes = False

    # timestamp = "2023-07-26_101624"
    print(timestamp)

    default_params_registration = "ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                                  "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 15000 " \
                                  "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis tensorboard"
    default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.5 --max-angle-factor 0.25 --scene-scale 2 --registration True " \
                                         "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False"

    print(scene_names)
    print(exp_types)

    exps = []

    # llff_scene_names = ["fern", "flower", "fortress", "horns", "orchids", "room", "trex"]
    # llff_scene_names = ["fern", "fortress", "horns", "room"]
    # scene_names = ["fern"]

    for scene in scene_names:
        for exp_type in exp_types:
            exp_params = {
                "data1": f"{data_dir}/{scene}/transforms_{exp_type}_1.json",
                "data2": f"{data_dir}/{scene}/transforms_{exp_type}_2.json",
                "inerf_data": f"{data_dir}/{scene}/inerf_{exp_type}_transforms.json",
                "experiment_name": f"{scene}_{exp_type}",
                "scene_name": f"{scene}",
                "downscale_factor": f"{downscale}",
                "num_points_reg": "25",
                "num_points_unreg": "10",
                "pretrain-iters": "25",
                "unreg_data_dir": f"{data_dir}/",
                "outputs_dir": f"{outputs_dir}"
            }
            exps.append(exp_params)

    total_stats = {}
    for exp in exps:
        print("experiment_name:", exp["experiment_name"])
        registeration_cmd = default_params_registration + " --output-dir " + exp["outputs_dir"] + \
                            " --pretrain-iters " + exp["pretrain-iters"] + " --machine.seed {}" \
                            + " --pipeline.datamanager.train-num-images-to-sample-from 1" \
                            + " --data " + exp["inerf_data"] \
                            + " --experiment_name " + exp[
                                "experiment_name"] + "_registration_inerf --timestamp " + timestamp \
                            + "_registration --timestamp " + timestamp \
                            + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                            + default_params_registration_suffix + " --downscale_factor " + exp["downscale_factor"] \
                            + " --inerf True" \
                            + " --registration_data " + outputs_dir + exp[
                                "experiment_name"] + "_registered/nerfacto/" + timestamp

        scene_seed = np.array(list(exp["scene_name"].encode('ascii'))).sum()

        best_psnr = 0
        for i in range(1, 11):
            os.system(registeration_cmd.format(str(scene_seed*i)))

            # Read the stats of the registration
            exp_stats_path = outputs_dir + exp[
                "experiment_name"] + "_registration_inerf/nerfacto/" + timestamp + "/stats.json"
            with open(os.path.join(exp_stats_path), 'r') as f:
                exp_stats = json.load(f)
            if exp_stats["psnr"] > best_psnr:
                best_psnr = exp_stats["psnr"]
                best_exp_stats = exp_stats

        total_stats[exp["experiment_name"]] = best_exp_stats

    curr_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    base_dir = f"{outputs_dir}/../stats/"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    total_stats_path = base_dir + curr_timestamp + "_inerf_init.json"
    with open(total_stats_path, "w") as outfile:
        json.dump(total_stats, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 6 and len(sys.argv) != 7:
        print("Usage: python reg_pipeline.py <data_directory> <output_directory> <scene_names> <exp_types> <downscale> <<timestamp>>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        scene_names = sys.argv[3].split(',')
        exp_types = sys.argv[4].split(',')
        downscale = sys.argv[5]

        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            if len(sys.argv) == 7:
                timestamp = sys.argv[6]
                main(base_directory, output_directory, scene_names, exp_types, downscale, timestamp)
            else:
                main(base_directory, output_directory, scene_names, exp_types, downscale)




