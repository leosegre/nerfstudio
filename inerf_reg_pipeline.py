import os
import subprocess
from datetime import datetime
import json

# timestamp = "2023-07-26_101624"
# timestamp = "2023-07-30_124120"
timestamp = "2023-08-13_180819"

print(timestamp)

outputs_dir = "/home/leo/nerfstudio_reg/nerfstudio/outputs/"
default_params = "ns-train nerfacto --viewer.quit-on-train-completion True --max-num-iterations 60000 --nf-first-iter 30000 --pipeline.datamanager.train-num-rays-per-batch 1024 " \
                 "--pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard "
default_params_registered = " nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 "
default_params_unregistered = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --registration True "
default_params_registration = "ns-train register-nerfacto-inerf --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                              "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 10000 --pipeline.datamanager.camera-optimizer.mode SE3 " \
                              "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard"
default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --registration True " \
                                     "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False"

exp_types = ["0_100_even_odd", "30_70_even_odd", "50_50"]
# exp_types = ["0_100_even_odd"]

exps = []
my_scene_names = ["lion", "table"]
for scene in my_scene_names:
    for exp_type in exp_types:
        exp_params = {
            "data1": f"/home/leo/data/{scene}/transforms_{exp_type}_1.json",
            "data2": f"/home/leo/data/{scene}/transforms_{exp_type}_2.json",
            "inerf_data": f"/home/leo/data/{scene}/inerf_{exp_type}_transforms.json",
            "experiment_name": f"{scene}_{exp_type}",
            "downscale_factor": "2",
            "num_points": "20",
            "pretrain-iters": "20",
            "unreg_data_dir": "/home/leo/data/"
        }
        # exps.append(exp_params)

# llff_scene_names = ["fern", "flower", "fortress", "horns", "orchids", "room", "trex"]
llff_scene_names = ["fern", "fortress", "horns", "room"]


for scene in llff_scene_names:
    for exp_type in exp_types:
        exp_params = {
            "data1": f"/home/leo/data/{scene}/transforms_{exp_type}_1.json",
            "data2": f"/home/leo/data/{scene}/transforms_{exp_type}_2.json",
            "inerf_data": f"/home/leo/data/{scene}/inerf_{exp_type}_transforms.json",
            "experiment_name": f"{scene}_{exp_type}",
            "downscale_factor": "2",
            "num_points_reg": "5",
            "num_points_unreg": "5",
            "pretrain-iters": "20",
            "unreg_data_dir": "/home/leo/data/"
        }
        exps.append(exp_params)

total_stats = {}
for exp in exps:
    print("experiment_name:", exp["experiment_name"])
    registered_scene_cmd = default_params + \
                           "--data " + exp["data1"] + " --experiment_name " + exp["experiment_name"]  \
                           + "_registered --timestamp " + timestamp + default_params_registered + \
                           "--downscale_factor " + exp["downscale_factor"]

    unregistered_scene_cmd = default_params + \
                           "--data " + exp["data2"] + " --experiment_name " + exp["experiment_name"]  \
                           + "_unregistered --timestamp " + timestamp + default_params_unregistered + \
                           "--downscale_factor " + exp["downscale_factor"] + \
                           " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

    export_cmd_unreg = "ns-export nf-cameras --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                 + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                 + " --num_points " + exp["num_points_unreg"] + " --downscale_factor " + exp["downscale_factor"]

    export_cmd_reg = "ns-export nf-cameras --load-config " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" \
                 + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_registered" \
                 + " --num_points " + exp["num_points_reg"] + " --downscale_factor " + exp["downscale_factor"]

    t0_cmd = "python scripts/generate_t0_list.py " + exp["unreg_data_dir"] + exp["experiment_name"] + "_registered/transforms.json " \
             + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered/transforms.json " + exp["downscale_factor"]


    registeration_cmd = default_params_registration + " --pretrain-iters " + exp["pretrain-iters"] + \
                        " --data " + exp["inerf_data"] + " --machine.seed {}" \
                        + " --pipeline.datamanager.train-num-images-to-sample-from 1"\
                        + " --experiment_name " + exp["experiment_name"] + "_registration_inerf --timestamp " + timestamp \
                        + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                        + default_params_registration_suffix + " --downscale_factor " + exp["downscale_factor"] + " --inerf True" \
                        + " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

    # os.system(registered_scene_cmd)
    # os.system(unregistered_scene_cmd)
    # os.system(export_cmd)

    best_psnr = 0
    for i in range(3):
        os.system(registeration_cmd.format(str(i)))
        # import ipdb; ipdb.set_trace()
        # Read the stats of the registration
        exp_stats_path = outputs_dir + exp["experiment_name"] + "_registration_inerf/nerfacto/" + timestamp + "/stats.json"
        with open(os.path.join(exp_stats_path), 'r') as f:
            exp_stats = json.load(f)
        if exp_stats["psnr"] > best_psnr:
            best_psnr = exp_stats["psnr"]
            best_exp_stats = exp_stats

    total_stats[exp["experiment_name"]] = best_exp_stats

curr_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
base_dir = "/home/leo/nerfstudio_reg/nerfstudio/stats/"
total_stats_path = base_dir + curr_timestamp + "_inerf+init.json"
with open(total_stats_path, "w") as outfile:
    json.dump(total_stats, outfile, indent=2)


## iNeRF without init
exps = []
my_scene_names = ["lion", "table"]
for scene in my_scene_names:
    for exp_type in exp_types:
        exp_params = {
            "data1": f"/home/leo/data/{scene}/transforms_{exp_type}_1.json",
            "data2": f"/home/leo/data/{scene}/transforms_{exp_type}_2.json",
            "inerf_data": f"/home/leo/data/{scene}/inerf_{exp_type}_transforms.json",
            "experiment_name": f"{scene}_{exp_type}",
            "downscale_factor": "2",
            "num_points_reg": "5",
            "num_points_unreg": "5",
            "pretrain-iters": "0",
            "unreg_data_dir": "/home/leo/data/"
        }
        # exps.append(exp_params)


for scene in llff_scene_names:
    for exp_type in exp_types:
        exp_params = {
            "data1": f"/home/leo/data/{scene}/transforms_{exp_type}_1.json",
            "data2": f"/home/leo/data/{scene}/transforms_{exp_type}_2.json",
            "inerf_data": f"/home/leo/data/{scene}/inerf_{exp_type}_transforms.json",
            "experiment_name": f"{scene}_{exp_type}",
            "downscale_factor": "2",
            "num_points_reg": "5",
            "num_points_unreg": "5",
            "pretrain-iters": "0",
            "unreg_data_dir": "/home/leo/data/"
        }
        exps.append(exp_params)

total_stats = {}
for exp in exps:
    print("experiment_name:", exp["experiment_name"])
    registered_scene_cmd = default_params + \
                           "--data " + exp["data1"] + " --experiment_name " + exp["experiment_name"]  \
                           + "_registered --timestamp " + timestamp + default_params_registered + \
                           "--downscale_factor " + exp["downscale_factor"]

    unregistered_scene_cmd = default_params + \
                           "--data " + exp["data2"] + " --experiment_name " + exp["experiment_name"]  \
                           + "_unregistered --timestamp " + timestamp + default_params_unregistered + \
                           "--downscale_factor " + exp["downscale_factor"] + \
                           " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

    export_cmd = "ns-export nf-cameras --load-config " + outputs_dir + exp["experiment_name"] + "_unregistered/nerfacto/" \
                 + timestamp + "/config.yml" + " --output-dir " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                 + " --num_points " + exp["num_points_unreg"] + " --downscale_factor " + exp["downscale_factor"]

    registeration_cmd = default_params_registration + " --pretrain-iters " + exp["pretrain-iters"] + \
                        " --data " + exp["inerf_data"] + " --machine.seed {}" \
                        + " --pipeline.datamanager.train-num-images-to-sample-from 1"\
                        + " --experiment_name " + exp["experiment_name"] + "_registration_inerf_no_init --timestamp " + timestamp \
                        + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                        + default_params_registration_suffix + " --downscale_factor " + exp["downscale_factor"] + " --inerf True" \
                        + " --registration_data " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp

    # os.system(registered_scene_cmd)
    # os.system(unregistered_scene_cmd)
    # os.system(export_cmd)

    best_psnr = 0
    for i in range(3):
        os.system(registeration_cmd.format(str(i)))

        # Read the stats of the registration
        exp_stats_path = outputs_dir + exp["experiment_name"] + "_registration_inerf_no_init/nerfacto/" + timestamp + "/stats.json"
        with open(os.path.join(exp_stats_path), 'r') as f:
            exp_stats = json.load(f)
        if exp_stats["psnr"] > best_psnr:
            best_psnr = exp_stats["psnr"]
            best_exp_stats = exp_stats

    total_stats[exp["experiment_name"]] = best_exp_stats

base_dir = "/home/leo/nerfstudio_reg/nerfstudio/stats/"
total_stats_path = base_dir + curr_timestamp + "_inerf.json"
with open(total_stats_path, "w") as outfile:
    json.dump(total_stats, outfile, indent=2)