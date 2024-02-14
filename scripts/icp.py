import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import json
import open3d as o3d

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


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

    A_sampled= sample_points_uniformly(A, num_samples)
    B_sampled = sample_points_uniformly(B, num_samples)

    A_points = np.asarray(A_sampled.points)
    B_points = np.asarray(B_sampled.points)

    # Call icp function
    T, distances, i = icp(A_points, B_points, max_iterations=max_iterations)

    # Convert Open3D registration result to dictionary
    result_dict = {
        "transformation_matrix": T.tolist(),
        "distances": distances.tolist(),
        "num_iters": i
    }

    # Save results to JSON file
    with open(output_path, 'w') as json_file:
        json.dump(result_dict, json_file, indent=4)

    transformed_A = A.transform(T)
    output_path = Path(output_path)
    output_path = str(output_path.parent / "Transformed_A.ply")
    o3d.io.write_point_cloud(output_path, transformed_A)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ICP with ply point clouds')
    parser.add_argument('path_A', type=str, help='Path to ply file A')
    parser.add_argument('path_B', type=str, help='Path to ply file B')
    parser.add_argument('output_path', type=str, help='Output path for JSON file')
    parser.add_argument('--max_iterations', type=int, default=20, help='Maximum iterations (default: 20)')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples from each PC (default: 100000)')
    args = parser.parse_args()

    # Call main function with provided arguments
    main(args.path_A, args.path_B, args.output_path, args.max_iterations, args.num_samples)
