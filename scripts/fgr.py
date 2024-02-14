import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import json
import open3d as o3d
from nerfstudio.utils import poses as pose_utils
import torch
import copy


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold,
            iteration_number=64))
    return result


def sample_points_uniformly(point_cloud, num_samples):
    # Sample points uniformly from the point cloud
    indices = np.random.choice(np.arange(len(point_cloud.points)), num_samples, replace=False)
    sampled_points = np.asarray(point_cloud.points)[indices]
    sampled_point_cloud = o3d.geometry.PointCloud()
    sampled_point_cloud.points = o3d.utility.Vector3dVector(sampled_points)
    return sampled_point_cloud


def main(path_A, path_B, output_path, max_iterations, num_samples):
    # Read point cloud data from ply files
    A = o3d.io.read_point_cloud(path_A)
    B = o3d.io.read_point_cloud(path_B)

    # A_sampled = sample_points_uniformly(A, num_samples)
    # B_sampled = sample_points_uniformly(B, num_samples)

    # A_points = np.asarray(A_sampled.points)
    # B_points = np.asarray(B_sampled.points)

    # Call FGR function
    # Run Fast Global Registration (FGR) to register the point clouds
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(copy.deepcopy(A),
                                                                                         copy.deepcopy(B), voxel_size)

    result = execute_fast_global_registration(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              voxel_size)
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                             source_fpfh, target_fpfh,
    #                                             voxel_size)

    # threshold = 0.02
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, threshold, result.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    # # colored pointcloud registration
    # # This is implementation of following paper
    # # J. Park, Q.-Y. Zhou, V. Koltun,
    # # Colored Point Cloud Registration Revisited, ICCV 2017
    # voxel_radius = [0.04, 0.02, 0.01]
    # max_iter = [50, 30, 14]
    # # current_transformation = np.identity(4)
    # current_transformation = result.transformation
    # print("3. Colored point cloud registration")
    # for scale in range(3):
    #     iter = max_iter[scale]
    #     radius = voxel_radius[scale]
    #     print([iter, radius, scale])
    #
    #     print("3-1. Downsample with a voxel size %.2f" % radius)
    #     source_down = source.voxel_down_sample(radius)
    #     target_down = target.voxel_down_sample(radius)
    #
    #     print("3-2. Estimate normal.")
    #     source_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #     target_down.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    #
    #     print("3-3. Applying colored point cloud registration")
    #     result_icp = o3d.pipelines.registration.registration_colored_icp(
    #         source_down, target_down, radius, current_transformation,
    #         o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #         o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                           relative_rmse=1e-6,
    #                                                           max_iteration=iter))
    #     # current_transformation = result_icp.transformation

    # Convert Open3D registration result to dictionary

    registration_matrix = result.transformation

    result_dict = {
        "t0_matrix": registration_matrix.tolist(),
    }

    # Save results to JSON file
    with open(output_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    transformed_A = copy.deepcopy(A).transform(registration_matrix)
    # transformed_B = copy.deepcopy(B).transform(registration_matrix)
    output_path = Path(output_path)
    output_path_A = str(output_path.parent / "Transformed_A.ply")
    # output_path_B = str(output_path.parent / "Transformed_B.ply")
    o3d.io.write_point_cloud(output_path_A, transformed_A)
    # o3d.io.write_point_cloud(output_path_B, transformed_B)
    # o3d.visualization.draw_geometries([transformed_A, B])


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ICP with ply point clouds')
    parser.add_argument('path_A', type=str, help='Path to ply file A')
    parser.add_argument('path_B', type=str, help='Path to ply file B')
    parser.add_argument('output_path', type=str, help='Output path for JSON file')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations (default: 20)')
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples from each PC (default: 100000)')
    args = parser.parse_args()

    # Call main function with provided arguments
    main(args.path_A, args.path_B, args.output_path, args.max_iterations, args.num_samples)
