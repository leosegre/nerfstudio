import torch
import torch.nn as nn
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RaySamples, RayBundle, Frustums
from pathlib import Path
from nerfstudio.fields.base_field import shift_directions_for_tcnn
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp

IN_CKPT_FILE = Path("/storage/leo/outputs/fern_0_100_even_odd_registered/nerfacto/2023-09-10_211852/config.yml")
OUT_PT_FILE = "./pt_file.pt"

load_path = IN_CKPT_FILE
config, pipeline, checkpoint_path, step = eval_setup(load_path)
print(f":white_check_mark: Done loading checkpoint from {load_path}")
print(pipeline)

ray_bundle, _ = pipeline.datamanager.next_train(0)
if pipeline.model.collider is not None:
    ray_bundle = pipeline.model.collider(ray_bundle)

ray_samples, weights_list, ray_samples_list = pipeline.model.proposal_sampler(ray_bundle, density_fns=pipeline.model.density_fns)

field = pipeline.model.field

torch.save(pipeline.model, OUT_PT_FILE)
nerf
# """Computes and returns the densities."""
# if pipeline.model.field.spatial_distortion is not None:
#     positions = ray_samples.frustums.get_positions()
#     positions = pipeline.model.field.spatial_distortion(positions)
#     positions = (positions + 2.0) / 4.0
# else:
#     positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), pipeline.model.aabb)
# # Make sure the tcnn gets inputs between 0 and 1.
# selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
# positions = positions * selector[..., None]
# pipeline.model.field._sample_locations = positions
# if not pipeline.model.field._sample_locations.requires_grad:
#     pipeline.model.field._sample_locations.requires_grad = True
# positions_flat = positions.view(-1, 3)
# h = pipeline.model.field.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
# density_before_activation, base_mlp_out = torch.split(h, [1, pipeline.model.field.geo_feat_dim], dim=-1)
# pipeline.model.field._density_before_activation = density_before_activation
#
# # Rectifying the density with an exponential is much more stable than a ReLU or
# # softplus, because it enables high post-activation (float32) density outputs
# # from smaller internal (float16) parameters.
# density = trunc_exp(density_before_activation.to(positions))
# density = density * selector[..., None]
# # return density, base_mlp_out
# density_embedding = base_mlp_out
#
# assert density_embedding is not None
# outputs = {}
# if ray_samples.camera_indices is None:
#     raise AttributeError("Camera indices are not provided.")
# camera_indices = ray_samples.camera_indices.squeeze()
# directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
# directions_flat = directions.view(-1, 3)
# d = pipeline.model.field.direction_encoding(directions_flat)
#
# outputs_shape = ray_samples.frustums.directions.shape[:-1]
#
# h = torch.cat(
#     [
#         d,
#         density_embedding.view(-1, pipeline.model.field.geo_feat_dim)
#     ],
#     dim=-1,
# )
# rgb = pipeline.model.field.mlp_head(h).view(*outputs_shape, -1).to(directions)
#
# # Trace a specific method and construct `ScriptModule` with
# # a single `forward` method
# module = torch.jit.trace(pipeline.model.field.mlp_base, torch.rand_like(positions_flat))
# module = torch.jit.trace(pipeline.model.field.mlp_head, torch.rand_like(h))
#
# # model = pipeline.model.field
# # m = torch.jit.script(model)
# # torch.hit.save(m, OUT_PT_FILE)
#
# # Save to .pt
# module.save(OUT_PT_FILE)