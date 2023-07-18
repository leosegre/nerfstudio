timestamp="$(date +"%Y-%m-%d_%H-%M-%S")"
echo $timestamp


ns-train nerfacto --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-rays-per-batch 1024 --pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --data /home/leo/data/table/transforms1.json --experiment_name table_nf_registered --timestamp $timestamp nerfstudio-data --auto_scale_poses True --train-split-fraction 1.0 --center-method focus --orientation-method up --scene-scale 2 --downscale_factor 1

ns-train nerfacto --viewer.quit-on-train-completion True --pipeline.datamanager.train-num-rays-per-batch 1024 --pipeline.model.predict-view-likelihood True --pipeline.datamanager.camera-optimizer.mode off --vis tensorboard --data /home/leo/data/table/transforms2.json --experiment_name table_nf_unregistered --timestamp $timestamp nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --downscale_factor 1 --registration True --registration_data /home/leo/nerfstudio_reg/nerfstudio/outputs/table_nf_registered/nerfacto/$timestamp/

ns-export nf-cameras --load-config outputs/table_nf_unregistered/nerfacto/$timestamp/config.yml --output-dir /home/leo/data/table_nf_unregistered/ --num_points 10 --depth 0.5 --downscale-factor 1

ns-train register-nerfacto --viewer.quit-on-train-completion True --pipeline.model.predict-view-likelihood True --nf-first-iter 100000 --pretrain-iters 20 --start-step 0 --pipeline.datamanager.train-num-rays-per-batch 32000 --max-num-iterations 10000 --pipeline.model.distortion-loss-mult 0 --pipeline.model.interlevel-loss-mult 0 --pipeline.registration True --vis viewer+tensorboard --data /home/leo/data/table_nf_unregistered/ --experiment_name table_nf_registration --timestamp $timestamp --load_dir /home/leo/nerfstudio_reg/nerfstudio/outputs/table_nf_registered/nerfacto/$timestamp/nerfstudio_models/ nerfstudio-data --train-split-fraction 1.0 --max-translation 0.25 --max-angle-factor 0.25 --scene-scale 2 --downscale_factor 1 --registration True --optimize_camera_registration True --load_registration True --orientation-method none --center-method none --auto-scale-poses False

