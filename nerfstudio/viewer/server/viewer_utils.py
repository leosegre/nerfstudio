# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-many-lines

"""Code to interface with the `vis/` (the JS viewer)."""
from __future__ import annotations

import enum
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.utils import colormaps
from nerfstudio.utils.io import load_from_json
from nerfstudio.viewer.server.control_panel import ControlPanel

from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3
import nerfstudio.utils.poses as pose_utils


warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer

CONSOLE = Console(width=120)


def get_viewer_version() -> str:
    """Return the version of the viewer."""
    json_filename = os.path.join(os.path.dirname(__file__), "../app/package.json")
    version = load_from_json(Path(json_filename))["version"]
    return version


def get_viewer_url(websocket_port: int) -> str:
    """Generate URL for the viewer.

    Args:
        websocket_port: port to connect to the viewer
    Returns:
        URL to the viewer
    """
    version = get_viewer_version()
    websocket_url = f"ws://localhost:{websocket_port}"
    return f"https://viewer.nerf.studio/versions/{version}/?websocket_url={websocket_url}"


class ColormapTypes(str, enum.Enum):
    """List of colormap render types"""

    DEFAULT = "default"
    TURBO = "turbo"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"


class IOChangeException(Exception):
    """Basic camera exception to interrupt viewer"""


class SetTrace:
    """Basic trace function"""

    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)


def is_port_open(port: int):
    """Returns True if the port is open.

    Args:
        port: Port to check.

    Returns:
        True if the port is open, False otherwise.
    """
    try:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        _ = sock.bind(("", port))
        sock.close()
        return True
    except OSError:
        return False


def get_free_port(default_port: Optional[int] = None):
    """Returns a free port on the local machine. Try to use default_port if possible.

    Args:
        default_port: Port to try to use.

    Returns:
        A free port on the local machine.
    """
    if default_port is not None:
        if is_port_open(default_port):
            return default_port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    return port


def update_render_aabb(
    crop_viewport: bool, crop_min: Tuple[float, float, float], crop_max: Tuple[float, float, float], model: Model
):
    """
    update the render aabb box for the viewer:

    Args:
        crop_viewport: whether to crop the viewport
        crop_min: min of the crop box
        crop_max: max of the crop box
        model: the model to render
    """

    if crop_viewport:
        crop_min = torch.tensor(crop_min, dtype=torch.float32)
        crop_max = torch.tensor(crop_max, dtype=torch.float32)

        if isinstance(model.render_aabb, SceneBox):
            model.render_aabb.aabb[0] = crop_min
            model.render_aabb.aabb[1] = crop_max
        else:
            model.render_aabb = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
    else:
        model.render_aabb = None


    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(self, dataset: InputDataset, start_train=True, registration=False) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            start_train: whether to start train when viewer init;
                if False, only displays dataset until resume train is toggled
        """
        # set the config base dir
        self.vis["renderingState/config_base_dir"].write(str(self.log_filename.parents[0]))

        # set the data base dir
        self.vis["renderingState/data_base_dir"].write(str(self.datapath))

        # set default export path name
        self.vis["renderingState/export_path"].write(self.log_filename.parent.stem)

        # clear the current scene
        self.vis["sceneState/sceneBox"].delete()
        self.vis["sceneState/cameras"].delete()

        # draw the training cameras and images
        image_indices = self._pick_drawn_image_idxs(len(dataset))
        for idx in image_indices:
            image = dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.vis[f"sceneState/cameras/{idx:06d}"].write(camera_json)
            if registration:
                # inv_unregistration_matrix = torch.inverse(dataset.metadata["unregistration_matrix"])
                registration_matrix = dataset.metadata["registration_matrix"]
                camera_to_world_tensor = torch.cat((torch.from_numpy(np.array(camera_json["camera_to_world"], dtype=np.float32)), \
                                                    torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), 0)
                # camera_json["camera_to_world"] = (inv_unregistration_matrix @ camera_to_world_tensor)[:3, :].tolist()
                camera_json["camera_to_world"] = pose_utils.multiply(registration_matrix, camera_to_world_tensor).tolist()
                self.vis[f"sceneState/cameras/{idx:06d}_original"].write(camera_json)

        # draw the scene box (i.e., the bounding box)
        json_ = dataset.scene_box.to_json()
        self.vis["sceneState/sceneBox"].write(json_)

        # set the initial state whether to train or not
        self.vis["renderingState/isTraining"].write(start_train)

        max_scene_box = torch.max(dataset.scene_box.aabb[1] - dataset.scene_box.aabb[0]).item()
        self.vis["renderingState/max_box_size"].write(max_scene_box)

    def update_register_cameras(self, datamanager: VanillaDataManager, step, pre_train=False) -> None:
        """Draw new register images in the viewer.

        Args:
            datamanager: datamanager to render in the scene
        """
        # draw the new training cameras and images
        image_indices = self._pick_drawn_image_idxs(len(datamanager.train_dataset))
        camera_opt_to_camera = datamanager.train_camera_optimizer([0])
        for idx in image_indices:
            image = datamanager.train_dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = datamanager.train_dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            # camera_opt_param_group = datamanager.train_camera_optimizer
            # camera_opt_params = datamanager.get_param_groups()[camera_opt_param_group][0].data
            # # Apply learned transformation delta.
            # if self.config.datamanager.camera_optimizer.mode == "off":
            #     pass
            # elif self.config.datamanager.camera_optimizer.mode == "SO3xR3":
            #     camera_opt_transform_matrix = exp_map_SO3xR3(camera_opt_params)
            # elif self.config.datamanager.camera_optimizer.mode == "SE3":
            #     camera_opt_transform_matrix = exp_map_SE3(camera_opt_params)
            #
            # camera_to_world_tensor = torch.cat((torch.from_numpy(np.array(camera_json["camera_to_world"], dtype=np.float32)), \
            #                                     torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), 0)
            # camera_json["camera_to_world"] = (camera_opt_transform_matrix @ camera_to_world_tensor)[:3, :].tolist()
            camera_to_world_tensor = torch.from_numpy(np.array(camera_json["camera_to_world"])).unsqueeze(0).to(device=camera_opt_to_camera.device, dtype=torch.float32)
            camera_json["camera_to_world"] = pose_utils.multiply(camera_opt_to_camera, camera_to_world_tensor).squeeze().tolist()
            # print(camera_json["camera_to_world"])
            if pre_train:
                self.vis[f"sceneState/cameras/{idx:06d}_step_pre_train{step:06d}"].write(camera_json)
            else:
                self.vis[f"sceneState/cameras/{idx:06d}_step_{step:06d}"].write(camera_json)

    def _check_camera_path_payload(self, trainer, step: int):
        """Check to see if the camera path export button was pressed."""
        # check if we should interrupt from a button press?
        camera_path_payload = self.vis["camera_path_payload"].read()
        if camera_path_payload:
            # save a model checkpoint
            trainer.save_checkpoint(step)
            # write to json file in datapath directory
            camera_path_filename = camera_path_payload["camera_path_filename"] + ".json"
            camera_path = camera_path_payload["camera_path"]
            camera_paths_directory = os.path.join(self.datapath, "camera_paths")
            if not os.path.exists(camera_paths_directory):
                os.mkdir(camera_paths_directory)

            write_to_json(Path(os.path.join(camera_paths_directory, camera_path_filename)), camera_path)
            self.vis["camera_path_payload"].delete()

    def _check_populate_paths_payload(self, trainer, step: int):
        populate_paths_payload = self.vis["populate_paths_payload"].read()
        if populate_paths_payload:
            # save a model checkpoint
            trainer.save_checkpoint(step)
            # get all camera paths
            camera_path_dir = os.path.join(self.datapath, "camera_paths")
            if os.path.exists(camera_path_dir):
                camera_path_files = os.listdir(camera_path_dir)
                all_path_dict = {}
                for i in camera_path_files:
                    if i[-4:] == "json":
                        all_path_dict[i[:-5]] = load_from_json(Path(os.path.join(camera_path_dir, i)))
                self.vis["renderingState/all_camera_paths"].write(all_path_dict)
                self.vis["populate_paths_payload"].delete()

    def _update_render_aabb(self, graph):
        """
        update the render aabb box for the viewer:

        :param graph:
        :return:
        """

        crop_enabled = self.vis["renderingState/crop_enabled"].read()
        if crop_enabled != self.prev_crop_enabled:
            self.camera_moving = True
            self.prev_crop_enabled = crop_enabled
            self.prev_crop_bg_color = None
            self.prev_crop_scale = None
            self.prev_crop_center = None

        if crop_enabled:
            crop_scale = self.vis["renderingState/crop_scale"].read()
            crop_center = self.vis["renderingState/crop_center"].read()
            crop_bg_color = self.vis["renderingState/crop_bg_color"].read()

            if crop_bg_color != self.prev_crop_bg_color:
                self.camera_moving = True
                self.prev_crop_bg_color = crop_bg_color

            if crop_scale != self.prev_crop_scale or crop_center != self.prev_crop_center:
                self.camera_moving = True
                self.prev_crop_scale = crop_scale
                self.prev_crop_center = crop_center

                crop_scale = torch.tensor(crop_scale)
                crop_center = torch.tensor(crop_center)

                box_min = crop_center - crop_scale / 2.0
                box_max = crop_center + crop_scale / 2.0

                if isinstance(graph.render_aabb, SceneBox):
                    graph.render_aabb.aabb[0] = box_min
                    graph.render_aabb.aabb[1] = box_max
                else:
                    graph.render_aabb = SceneBox(aabb=torch.stack([box_min, box_max], dim=0))

                # maybe should update only if true change ?
                json_ = graph.render_aabb.to_json()
                self.vis["sceneState/sceneBox"].write(json_)
        else:
            graph.render_aabb = None

    def update_scene(self, trainer, step: int, graph: Model, num_rays_per_batch: int) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            graph: the current checkpoint of the model
        """
        has_temporal_distortion = getattr(graph, "temporal_distortion", None) is not None
        self.vis["model/has_temporal_distortion"].write(str(has_temporal_distortion).lower())

        is_training = self.vis["renderingState/isTraining"].read()
        self.step = step

        self._check_camera_path_payload(trainer, step)
        self._check_populate_paths_payload(trainer, step)

        camera_object = self._get_camera_object()
        if camera_object is None:
            return

        if is_training is None or is_training:
            # in training mode

            if self.camera_moving:
                # if the camera is moving, then we pause training and update camera continuously

                while self.camera_moving:
                    self._render_image_in_viewer(camera_object, graph, is_training)
                    camera_object = self._get_camera_object()
            else:
                # if the camera is not moving, then we approximate how many training steps need to be taken
                # to render at a FPS defined by self.static_fps.

                if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                    train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                    target_train_util = self.vis["renderingState/targetTrainUtil"].read()
                    if target_train_util is None:
                        target_train_util = 0.9

                    batches_per_sec = train_rays_per_sec / num_rays_per_batch

                    num_steps = max(int(1 / self.static_fps * batches_per_sec), 1)
                else:
                    num_steps = 1

                if step % num_steps == 0:
                    self._render_image_in_viewer(camera_object, graph, is_training)

        else:
            # in pause training mode, enter render loop with set graph
            local_step = step
            run_loop = not is_training
            while run_loop:
                # if self._is_render_step(local_step) and step > 0:
                if step > 0:
                    self._render_image_in_viewer(camera_object, graph, is_training)
                    camera_object = self._get_camera_object()
                is_training = self.vis["renderingState/isTraining"].read()
                self._check_populate_paths_payload(trainer, step)
                self._check_camera_path_payload(trainer, step)
                run_loop = not is_training
                local_step += 1

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.check_interrupt_vis and not self.camera_moving:
                raise IOChangeException
        return self.check_interrupt

    def _get_camera_object(self):
        """Gets the camera object from the viewer and updates the movement state if it has changed."""

        data = self.vis["renderingState/camera"].read()
        if data is None:
            return None

        camera_object = data["object"]
        render_time = self.vis["renderingState/render_time"].read()

        if render_time is not None:
            if (
                self.prev_camera_matrix is not None and np.allclose(camera_object["matrix"], self.prev_camera_matrix)
            ) and (self.prev_render_time == render_time):
                self.camera_moving = False
            else:
                self.prev_camera_matrix = camera_object["matrix"]
                self.prev_render_time = render_time
                self.camera_moving = True
        else:
            if self.prev_camera_matrix is not None and np.allclose(camera_object["matrix"], self.prev_camera_matrix):
                self.camera_moving = False
            else:
                self.prev_camera_matrix = camera_object["matrix"]
                self.camera_moving = True

        output_type = self.vis["renderingState/output_choice"].read()
        if output_type is None:
            output_type = OutputTypes.INIT
        if self.prev_output_type != output_type:
            self.camera_moving = True

        colormap_type = self.vis["renderingState/colormap_choice"].read()
        if self.prev_colormap_type != colormap_type:
            self.camera_moving = True

        colormap_range = self.vis["renderingState/colormap_range"].read()
        if self.prev_colormap_range != colormap_range:
            self.camera_moving = True

        colormap_invert = self.vis["renderingState/colormap_invert"].read()
        if self.prev_colormap_invert != colormap_invert:
            self.camera_moving = True

        colormap_normalize = self.vis["renderingState/colormap_normalize"].read()
        if self.prev_colormap_normalize != colormap_normalize:
            self.camera_moving = True

        crop_bg_color = self.vis["renderingState/crop_bg_color"].read()
        if self.prev_crop_enabled:
            if self.prev_crop_bg_color != crop_bg_color:
                self.camera_moving = True

        crop_scale = self.vis["renderingState/crop_scale"].read()
        if self.prev_crop_enabled:
            if self.prev_crop_scale != crop_scale:
                self.camera_moving = True

        crop_center = self.vis["renderingState/crop_center"].read()
        if self.prev_crop_enabled:
            if self.prev_crop_center != crop_center:
                self.camera_moving = True

        return camera_object

    # def _apply_colormap(self, outputs: Dict[str, Any], colors: torch.Tensor = None, eps=1e-6):
    #     """Determines which colormap to use based on set colormap type

    #     Args:
    #         outputs: the output tensors for which to apply colormaps on
    #         colors: is only set if colormap is for semantics. Defaults to None.
    #         eps: epsilon to handle floating point comparisons
    #     """
    #     if self.output_list:
    #         reformatted_output = self._process_invalid_output(self.prev_output_type)

    #     # default for rgb images
    #     if self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].shape[-1] == 3:
    #         return outputs[reformatted_output]

    #     # rendering depth outputs
    #     if outputs[reformatted_output].shape[-1] == 1 and outputs[reformatted_output].dtype == torch.float:
    #         output = outputs[reformatted_output]
    #         if self.prev_colormap_normalize:
    #             output = output - torch.min(output)
    #             output = output / (torch.max(output) + eps)
    #         output = output * (self.prev_colormap_range[1] - self.prev_colormap_range[0]) + self.prev_colormap_range[0]
    #         output = torch.clip(output, 0, 1)
    #         if self.prev_colormap_invert:
    #             output = 1 - output
    #         if self.prev_colormap_type == ColormapTypes.DEFAULT:
    #             return colormaps.apply_colormap(output, cmap=ColormapTypes.TURBO.value)
    #         return colormaps.apply_colormap(output, cmap=self.prev_colormap_type)

    #     # rendering semantic outputs
    #     if outputs[reformatted_output].dtype == torch.int:
    #         logits = outputs[reformatted_output]
    #         labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)  # type: ignore
    #         assert colors is not None
    #         return colors[labels]

    #     # rendering boolean outputs
    #     if outputs[reformatted_output].dtype == torch.bool:
    #         return colormaps.apply_boolean_colormap(outputs[reformatted_output])

    #     raise NotImplementedError

    def _send_output_to_viewer(self, outputs: Dict[str, Any], colors: torch.Tensor = None):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the graph
            colors: is only set if colormap is for semantics. Defaults to None.
        """
        if self.output_list is None:
            self.output_list = list(outputs.keys())
            viewer_output_list = list(np.copy(self.output_list))
            # remapping rgb_fine -> rgb for all cases just so that we dont have 2 of them in the options
            if OutputTypes.RGB_FINE in self.output_list:
                viewer_output_list.remove(OutputTypes.RGB_FINE)
            viewer_output_list.insert(0, OutputTypes.RGB)
            # remove semantics, which crashes viewer; semantics_colormap is OK
            if "semantics" in self.output_list:
                viewer_output_list.remove("semantics")
            self.vis["renderingState/output_options"].write(viewer_output_list)

        reformatted_output = self._process_invalid_output(self.prev_output_type)
        # re-register colormaps and send to viewer
        if self.output_type_changed or self.prev_colormap_type is None:
            self.prev_colormap_type = ColormapTypes.DEFAULT
            colormap_options = []
            self.vis["renderingState/colormap_options"].write(list(ColormapTypes))
            if outputs[reformatted_output].shape[-1] == 3:
                colormap_options = [ColormapTypes.DEFAULT]
            if outputs[reformatted_output].shape[-1] == 1 and outputs[reformatted_output].dtype == torch.float:
                self.prev_colormap_type = ColormapTypes.TURBO
                colormap_options = list(ColormapTypes)[1:]
            self.output_type_changed = False
            self.vis["renderingState/colormap_choice"].write(self.prev_colormap_type)
            self.vis["renderingState/colormap_options"].write(colormap_options)
        selected_output = (self._apply_colormap(outputs, colors) * 255).type(torch.uint8)

        image = selected_output[..., [2, 1, 0]].cpu().numpy()

        data = cv2.imencode(
            f".{self.config.image_format}",
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.config.jpeg_quality,
                cv2.IMWRITE_PNG_COMPRESSION,
                self.config.png_compression,
            ],
        )[1].tobytes()
        data = str(f"data:image/{self.config.image_format};base64," + base64.b64encode(data).decode("ascii"))
        self.vis["render_img"].write(data)

    def _update_viewer_stats(self, render_time: float, num_rays: int, image_height: int, image_width: int) -> None:
        """Function that calculates and populates all the rendering statistics accordingly

        Args:
            render_time: total time spent rendering current view
            num_rays: number of rays rendered
            image_height: resolution of the current view
            image_width: resolution of the current view
        """
        writer.put_time(
            name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=self.step, avg_over_steps=True
        )
        is_training = self.vis["renderingState/isTraining"].read()
        self.vis["renderingState/eval_res"].write(f"{image_height}x{image_width}px")
        if is_training is None or is_training:
            # process remaining training ETA
            self.vis["renderingState/train_eta"].write(GLOBAL_BUFFER["events"].get(EventName.ETA.value, "Starting"))
            # process ratio time spent on vis vs train
            if (
                EventName.ITER_VIS_TIME.value in GLOBAL_BUFFER["events"]
                and EventName.ITER_TRAIN_TIME.value in GLOBAL_BUFFER["events"]
            ):
                vis_time = GLOBAL_BUFFER["events"][EventName.ITER_VIS_TIME.value]["avg"]
                train_time = GLOBAL_BUFFER["events"][EventName.ITER_TRAIN_TIME.value]["avg"]
                vis_train_ratio = f"{int(vis_time / train_time * 100)}% spent on viewer"
                self.vis["renderingState/vis_train_ratio"].write(vis_train_ratio)
            else:
                self.vis["renderingState/vis_train_ratio"].write("Starting")
        else:
            self.vis["renderingState/train_eta"].write("Paused")
            self.vis["renderingState/vis_train_ratio"].write("100% spent on viewer")

    def _calculate_image_res(self, camera_object, is_training: bool) -> Optional[Tuple[int, int]]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            camera_object: the camera object to use for rendering
            is_training: whether or not we are training
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_resolution = self.vis["renderingState/maxResolution"].read()
        if max_resolution:
            self.max_resolution = max_resolution

        if self.camera_moving or not is_training:
            target_train_util = 0
        else:
            target_train_util = self.vis["renderingState/targetTrainUtil"].read()
            if target_train_util is None:
                target_train_util = 0.9

        if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
            train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
        elif not is_training:
            train_rays_per_sec = (
                80000  # TODO(eventually find a way to not hardcode. case where there are no prior training steps)
            )
        else:
            return None, None
        if EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
            vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
        else:
            vis_rays_per_sec = train_rays_per_sec

        current_fps = self.moving_fps if self.camera_moving else self.static_fps

        # calculate number of rays that can be rendered given the target fps
        num_vis_rays = vis_rays_per_sec / current_fps * (1 - target_train_util)

        aspect_ratio = camera_object["aspect"]

        if not self.camera_moving and not is_training:
            image_height = self.max_resolution
        else:
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(self.max_resolution, image_height), 30)
        image_width = int(image_height * aspect_ratio)
        if image_width > self.max_resolution:
            image_width = self.max_resolution
            image_height = int(image_width / aspect_ratio)
        return image_height, image_width

    def _process_invalid_output(self, output_type: str) -> str:
        """Check to see whether we are in the corner case of RGB; if still invalid, throw error
        Returns correct string mapping given improperly formatted output_type.

        Args:
            output_type: reformatted output type
        """
        if output_type == OutputTypes.INIT:
            output_type = OutputTypes.RGB

        # check if rgb or rgb_fine should be the case TODO: add other checks here
        attempted_output_type = output_type
        if output_type not in self.output_list and output_type == OutputTypes.RGB:
            output_type = OutputTypes.RGB_FINE

        # check if output_type is not in list
        if output_type not in self.output_list:
            assert (
                NotImplementedError
            ), f"Output {attempted_output_type} not in list. Tried to reformat as {output_type} but still not found."
        return output_type

    @profiler.time_function
    def _render_image_in_viewer(self, camera_object, graph: Model, is_training: bool) -> None:
        # pylint: disable=too-many-statements
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent over a TCP connection.

        Args:
            graph: current checkpoint of model
        """
        # Check that timestamp is newer than the last one
        if int(camera_object["timestamp"]) < self.prev_camera_timestamp:
            return

        self.prev_camera_timestamp = int(camera_object["timestamp"])

        # check and perform output type updates
        output_type = self.vis["renderingState/output_choice"].read()
        output_type = OutputTypes.INIT if output_type is None else output_type
        self.output_type_changed = self.prev_output_type != output_type
        self.prev_output_type = output_type

        # check and perform colormap type updates
        colormap_type = self.vis["renderingState/colormap_choice"].read()
        self.prev_colormap_type = colormap_type

        colormap_invert = self.vis["renderingState/colormap_invert"].read()
        self.prev_colormap_invert = colormap_invert

        colormap_normalize = self.vis["renderingState/colormap_normalize"].read()
        self.prev_colormap_normalize = colormap_normalize

        colormap_range = self.vis["renderingState/colormap_range"].read()
        self.prev_colormap_range = colormap_range

        # update render aabb
        try:
            self._update_render_aabb(graph)
        except RuntimeError as e:
            self.vis["renderingState/log_errors"].write("Got an Error while trying to update aabb crop")
            print(f"Error: {e}")

            time.sleep(0.5)  # sleep to allow buffer to reset

        # Calculate camera pose and intrinsics
        try:
            image_height, image_width = self._calculate_image_res(camera_object, is_training)
        except ZeroDivisionError as e:
            self.vis["renderingState/log_errors"].write("Error: Screen too small; no rays intersecting scene.")
            time.sleep(0.03)  # sleep to allow buffer to reset
            print(f"Error: {e}")
            return

        if image_height is None:
            return

        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            camera_object, image_height=image_height, image_width=image_width
        )

        camera_to_world = camera_to_world_h[:3, :]
        camera_to_world = torch.stack(
            [
                camera_to_world[0, :],
                camera_to_world[2, :],
                camera_to_world[1, :],
            ],
            dim=0,
        )

        camera_type_msg = camera_object["camera_type"]
        if camera_type_msg == "perspective":
            camera_type = CameraType.PERSPECTIVE
        elif camera_type_msg == "fisheye":
            camera_type = CameraType.FISHEYE
        elif camera_type_msg == "equirectangular":
            camera_type = CameraType.EQUIRECTANGULAR
        else:
            camera_type = CameraType.PERSPECTIVE

        camera = Cameras(
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
            camera_type=camera_type,
            camera_to_worlds=camera_to_world[None, ...],
            times=torch.tensor([float(self.prev_render_time)]),
        )
        camera = camera.to(graph.device)

        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=graph.render_aabb)

        graph.eval()

        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, graph=graph, camera_ray_bundle=camera_ray_bundle)

        check_thread.daemon = True
        render_thread.daemon = True

        with TimeWriter(None, None, write=False) as vis_t:
            check_thread.start()
            render_thread.start()
            try:
                render_thread.join()
                check_thread.join()
            except IOChangeException:
                del camera_ray_bundle
                torch.cuda.empty_cache()
            except RuntimeError as e:
                self.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )
                print(f"Error: {e}")
                del camera_ray_bundle
                torch.cuda.empty_cache()
                time.sleep(0.5)  # sleep to allow buffer to reset

        graph.train()
        outputs = render_thread.vis_outputs
        if outputs is not None:
            colors = graph.colors if hasattr(graph, "colors") else None
            self._send_output_to_viewer(outputs, colors=colors)
            self._update_viewer_stats(
                vis_t.duration, num_rays=len(camera_ray_bundle), image_height=image_height, image_width=image_width
            )


def apply_colormap(
    control_panel: ControlPanel, outputs: Dict[str, Any], colors: Optional[torch.Tensor] = None, eps=1e-6
):
    """Determines which colormap to use based on set colormap type

    Args:
        control_panel: control panel object
        outputs: the output tensors for which to apply colormaps on
        colors: is only set if colormap is for semantics. Defaults to None.
        eps: epsilon to handle floating point comparisons
    """
    colormap_type = control_panel.colormap
    output_type = control_panel.output_render

    # default for rgb images
    if colormap_type == ColormapTypes.DEFAULT and outputs[output_type].shape[-1] == 3:
        return outputs[output_type]

    # rendering depth outputs
    if outputs[output_type].shape[-1] == 1 and outputs[output_type].dtype == torch.float:
        output = outputs[output_type]
        if control_panel.colormap_normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = output * (control_panel.colormap_max - control_panel.colormap_min) + control_panel.colormap_min
        output = torch.clip(output, 0, 1)
        if control_panel.colormap_invert:
            output = 1 - output
        if colormap_type == ColormapTypes.DEFAULT:
            return colormaps.apply_colormap(output, cmap=ColormapTypes.TURBO.value)
        return colormaps.apply_colormap(output, cmap=colormap_type)

    # rendering semantic outputs
    if outputs[output_type].dtype == torch.int:
        logits = outputs[output_type]
        labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)  # type: ignore
        assert colors is not None
        return colors[labels]

    # rendering boolean outputs
    if outputs[output_type].dtype == torch.bool:
        return colormaps.apply_boolean_colormap(outputs[output_type])

    raise NotImplementedError
