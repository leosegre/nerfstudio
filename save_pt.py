import torch
import torch.nn as nn
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RaySamples, RayBundle, Frustums
from pathlib import Path

IN_CKPT_FILE = Path("/storage/leo/outputs/fern_0_100_even_odd_registered/nerfacto/2023-09-10_211852/config.yml")
OUT_PT_FILE = "./pt_file.pt"

load_path = IN_CKPT_FILE
config, pipeline, checkpoint_path, step = eval_setup(load_path)
print(f":white_check_mark: Done loading checkpoint from {load_path}")
print(pipeline)

ray_bundle, _ = pipeline.datamanager.next_train(0)
# ray_bundle = ray_bundle.get_row_major_sliced_ray_bundle(0, 1)

# Trace a specific method and construct `ScriptModule` with
# a single `forward` method
module = torch.jit.trace(pipeline.model, ray_bundle)

# Save to .pt
module.save(OUT_PT_FILE)