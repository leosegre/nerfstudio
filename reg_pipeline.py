import os
import subprocess
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
print(timestamp)

outputs_dir = "/home/leo/nerfstudio_reg/nerfstudio/outputs/"
default_params = "ns-train nerfacto --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-rays-per-batch 1024 " \
                 "--pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard "
default_params_registered = " nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 "
default_params_unregistered = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --registration True "
default_params_registration = "ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 " \
                              "--start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 10000 " \
                              "--pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard"
default_params_registration_suffix = " nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --registration True " \
                                     "--optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False"

exps = []

table_params = {
    "data1": "/home/leo/data/table/transforms1.json",
    "data2": "/home/leo/data/table/transforms2.json",
    "experiment_name": "table_nf",
    "downscale_factor": "1",
    "num_points": "10",
    "depth": "0.5",
    "pretrain-iters": "20",
    "unreg_data_dir": "/home/leo/data/"
}
# exps.append(table_params)
horns_params = {
    "data1": "/home/leo/data/horns/transforms1.json",
    "data2": "/home/leo/data/horns/transforms2.json",
    "experiment_name": "horns_nf",
    "downscale_factor": "2",
    "num_points": "10",
    "depth": "0.5",
    "pretrain-iters": "20",
    "unreg_data_dir": "/home/leo/data/"
}
exps.append(horns_params)



for exp in exps:
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
                 + " --num_points " + exp["num_points"] + " --depth " + exp["depth"] + " --downscale_factor " + exp["downscale_factor"]

    registeration_cmd = default_params_registration + " --pretrain-iters " + exp["pretrain-iters"] + \
                        " --data " + exp["unreg_data_dir"] + exp["experiment_name"] + "_unregistered" \
                        + " --experiment_name " + exp["experiment_name"] + "_registration --timestamp " + timestamp \
                        + " --load_dir " + outputs_dir + exp["experiment_name"] + "_registered/nerfacto/" + timestamp + "/nerfstudio_models/" \
                        + default_params_registration_suffix + " --downscale_factor " + exp["downscale_factor"]

    subprocess.run(registered_scene_cmd)
    subprocess.run(unregistered_scene_cmd)
    subprocess.run(export_cmd)
    subprocess.run(registeration_cmd)


# ns-train nerfacto --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-rays-per-batch 1024 --pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --data /home/leo/data/table/transforms1.json --experiment_name table_nf_registered --timestamp $timestamp nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 --downscale_factor 1
#
# ns-train nerfacto --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-rays-per-batch 1024 --pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --data /home/leo/data/table/transforms2.json --experiment_name table_nf_unregistered --timestamp $timestamp nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --downscale_factor 1 --registration True --registration_data /home/leo/nerfstudio_reg/nerfstudio/outputs/table_nf_registered/nerfacto/$timestamp/
#
# ns-export nf-cameras --load-config outputs/table_nf_unregistered/nerfacto/$timestamp/config.yml --output-dir /home/leo/data/table_nf_unregistered/ --num_points 10 --depth 0.5 --downscale-factor 1
#
# ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 --pretrain-iters 20 --start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 10000 --pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard --data /home/leo/data/table_nf_unregistered/ --experiment_name table_nf_registration --timestamp $timestamp --load_dir /home/leo/nerfstudio_reg/nerfstudio/outputs/table_nf_registered/nerfacto/$timestamp/nerfstudio_models/ nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --downscale_factor 1 --registration True --optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False

