import json
import sys
import numpy as np
from nerfstudio.cameras import camera_utils
from nerfstudio.utils import poses as pose_utils
import torch
def quaternion_look_at(direction, up):
    # Calculate the rotation matrix to look at the specified direction
    forward = direction / np.linalg.norm(direction)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up = new_up / np.linalg.norm(new_up)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = new_up
    rotation_matrix[:3, 2] = -forward

    # Convert the rotation matrix to a quaternion
    quaternion = np.zeros(4)
    t = np.trace(rotation_matrix) + 1.0

    if t > 1e-10:
        s = 0.5 / np.sqrt(t)
        quaternion[0] = 0.25 / s
        quaternion[1] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        quaternion[2] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        quaternion[3] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            quaternion[0] = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            quaternion[1] = 0.25 * s
            quaternion[2] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            quaternion[3] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            quaternion[0] = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            quaternion[1] = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            quaternion[2] = 0.25 * s
            quaternion[3] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            quaternion[0] = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            quaternion[1] = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            quaternion[2] = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            quaternion[3] = 0.25 * s

    return quaternion


def random_point_on_sphere(radius=1.0):
    # Generate a random point on the unit sphere
    phi = 2 * np.pi * np.random.random()
    costheta = np.random.random()
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.array([x, y, z])

def quaternion_to_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to 4x4 transformation matrix.
    """
    q = np.array([qw, qx, qy, qz])
    q = q / np.linalg.norm(q)  # Normalize quaternion

    # Extract quaternion components
    qw, qx, qy, qz = q

    # Compute transformation matrix
    matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2,   2*qx*qy - 2*qz*qw,   2*qx*qz + 2*qy*qw, 0],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw, 0],
        [2*qx*qz - 2*qy*qw,   2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2, 0],
        [0, 0, 0, 1]
    ])

    return matrix

def generate_cameras_on_sphere_towards_center(num_cameras, radius=1.0):
    cameras = []
    camera_to_world_list = []

    for i in range(num_cameras):
        # Generate a random point on the unit sphere
        random_point = torch.tensor(random_point_on_sphere(radius), dtype=torch.float64)

        # Calculate the quaternion to look at the center (0, 0, 0)
        direction = random_point
        up = torch.tensor(np.array([0.0, 0.0, 1.0]))  # Up vector for the camera
        # quaternion = quaternion_look_at(direction, up)

        # Convert quaternion to 4x4 transformation matrix
        # camera_to_world_matrix = quaternion_to_matrix(*quaternion)
        # camera_to_world_matrix[:3, 3] = random_point

        camera_to_world_matrix = camera_utils.viewmatrix(direction, up, random_point)
        camera_to_world_matrix = pose_utils.to4x4(camera_to_world_matrix)

        # Create a camera dictionary
        camera = {
            "matrix": str(camera_to_world_matrix.T.flatten().tolist()),
            "fov": 50,
            "aspect": 1,
            "properties": f"[[\"FOV\",50],[\"NAME\",\"Camera {i}\"],[\"TIME\",{i/float(num_cameras-1)}]]"
        }

        camera_to_world = {
            "camera_to_world": camera_to_world_matrix.flatten().tolist(),
            "fov": 50,
            "aspect": 1,
        }

        cameras.append(camera)
        camera_to_world_list.append(camera_to_world)

    return cameras, camera_to_world_list


def save_to_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <output_file_path>")
        sys.exit(1)

    output_file_path = sys.argv[1]

    # Generate 10 random cameras on the unit sphere towards the center
    random_cameras, c2w_list = generate_cameras_on_sphere_towards_center(10)

    # Create a JSON object with the cameras
    json_data = {
        "keyframes": random_cameras,
        "camera_type": "perspective",
        "render_height": 540,
        "render_width": 960,
        "camera_path": c2w_list,
        "fps": 1,
        "seconds": 9,
        "smoothness_value": 0.5,
        "is_cycle": False,
        "crop": None
    }

    # Save the JSON data to the specified file path
    save_to_json_file(json_data, output_file_path)

    print(f"JSON data saved to {output_file_path}")
