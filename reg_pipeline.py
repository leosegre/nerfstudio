import os
import subprocess
from datetime import datetime
import sys
import numpy as np

import json


def main(data_dir, outputs_dir, scene_names, exp_types, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        reconstruct_scenes = True
    else:
        reconstruct_scenes = False

    # timestamp = "2023-07-26_101624"
    print(timestamp)

    default_params = "ns-train nerfacto --viewer.quit-on-train-completion True --max-num-iterations 45000 --nf-first-iter 40000 --pipeline.datamanager.train-num-rays-per-batch 1024 " \
                     "--pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard "
    default_params_registered = " nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 "
    default_params_unregistered = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.5 --max-angle-factor 0.25 --scene-scale 2 " \
                                  "--registration True --orientation-method none --center-method none --auto-scale-poses False "
    default_params_registration = "ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                                  "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 10000 " \
                                  "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard"
    default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.5 --max-angle-factor 0.25 --scene-scale 2 --registration True " \
                                         "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False"

    # --pipeline.datamanager.camera-optimizer.mode SE3
    # exp_types = ["0_100_even_odd","30_70_even_odd","50_50"]
    # exp_types = ["50_50"]
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
                "experiment_name": f"{scene}_{exp_type}",
                "scene_name": f"{scene}",
                "downscale_factor": "2",
                "num_points_reg": "5",
                "num_points_unreg": "7",
                "pretrain-iters": "25",
                "unreg_data_dir": f"{data_dir}/",
                "outputs_dir": f"{outputs_dir}"
            }
            exps.append(exp_params)

    total_stats = {}
    for exp in exps:
        print("experiment_name:", exp["experiment_name"])
        registered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}"\
                               " --data " + exp["data1"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_registered --timestamp " + timestamp + default_params_registered + \
                               "--downscale_factor " + exp["downscale_factor"]

        unregistered_scene_cmd = default_params + "--output-dir " + exp["outputs_dir"] + " --machine.seed {}" \
                               " --data " + exp["data2"] + " --experiment_name " + exp["experiment_name"]  \
                               + "_unregistered --timestamp " + timestamp + default_params_unregistered + \
                               "--downscale_factor " + exp["downscale_factor"] + \
                               " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

        export_cmd_unreg = "ns-export nf-cameras --seed {} --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                     + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                     + " --num_points " + exp["num_points_unreg"] + " --downscale_factor " + exp["downscale_factor"]

        export_cmd_reg = "ns-export nf-cameras --load-config " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" \
                     + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_registered" \
                     + " --num_points " + exp["num_points_reg"] + " --downscale_factor " + exp["downscale_factor"]

        t0_cmd = "python scripts/generate_t0_list.py " + exp["unreg_data_dir"] + exp["experiment_name"] + "_registered/transforms.json " \
                 + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered/transforms.json " + exp["downscale_factor"]

        registeration_cmd = default_params_registration + " --output-dir " + exp["outputs_dir"] + \
                            " --pretrain-iters " + exp["pretrain-iters"] + " --machine.seed {}" \
                            + " --data " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                            + " --experiment_name " + exp["experiment_name"] + "_registration --timestamp " + timestamp \
                            + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                            + default_params_registration_suffix + " --downscale_factor " + exp["downscale_factor"]

        if reconstruct_scenes:
            scene_seed = np.array(list(exp["scene_name"].encode('ascii'))).sum()
            os.system(registered_scene_cmd.format(scene_seed))
            os.system(unregistered_scene_cmd.format(scene_seed))

        best_psnr = 0
        for i in range(10):
            os.system(export_cmd_unreg.format(str(i)))
            os.system(registeration_cmd.format(str(i)))

            # Read the stats of the registration
            exp_stats_path = outputs_dir + exp["experiment_name"] + "_registration/nerfacto/" + timestamp + "/stats.json"
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
    total_stats_path = base_dir + curr_timestamp + ".json"
    with open(total_stats_path, "w") as outfile:
        json.dump(total_stats, outfile, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: python reg_pipeline.py <data_directory> <output_directory> <scene_names> <exp_types> <<timestamp>>")
    else:
        base_directory = sys.argv[1]
        output_directory = sys.argv[2]
        scene_names = sys.argv[3].split(',')
        exp_types = sys.argv[4].split(',')
        if not os.path.isdir(base_directory):
            print(f"Error: {base_directory} is not a valid directory.")
        elif not os.path.isdir(output_directory):
            print(f"Error: {output_directory} is not a valid directory.")
        else:
            if len(sys.argv) == 6:
                timestamp = sys.argv[5]
                main(base_directory, output_directory, scene_names, exp_types, timestamp)
            else:
                main(base_directory, output_directory, scene_names, exp_types)




