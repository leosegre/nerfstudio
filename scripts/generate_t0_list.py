import sys
import shutil
import os
import json
import numpy as np
import cv2 as cv
import torch
from pathlib import Path
import nerfstudio.utils.poses as pose_utils
from scipy.spatial.transform import Rotation as R
from nerfstudio.cameras.lie_groups import exp_map_SE3, exp_map_SO3xR3



def get_homograph_loss(unreg_image, reg_image, unreg_frame, reg_frame, unreg_mask=None, reg_mask=None):
    rgb_loss = torch.nn.MSELoss()
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    MIN_MATCH_COUNT = 10

    # Calculate Homography
    sift = cv.SIFT_create()
    # print(unreg_image.shape)
    kp_image, des_image = sift.detectAndCompute(unreg_image, unreg_mask)
    kp_rgb, des_rgb = sift.detectAndCompute(reg_image, reg_mask)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)

    bf = cv.BFMatcher(crossCheck=True)

    if len(kp_rgb) == 0:
        matches = []
    else:
        matches = bf.knnMatch(des_image, des_rgb, k=1)

    # Need to draw only good matches, so create a mask
    good = []
    # ratio test as per Lowe's paper
    for i, m in enumerate(matches):
        if len(m) > 0:
            good.append(m[0])
    # print("step:", step, "good:", len(good))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 13.0)
        matchesMask = mask.ravel().tolist()
        if np.array(matchesMask).sum() == 0:
            print("matchesMask is 0 for in all entries")
            matchesMask = None
        else:
            h, w = unreg_image.shape[:-1]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            color_rgb_poly = cv.polylines(reg_image, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               matchesMask=None,
                               flags=2)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        src_pts, dst_pts = None, None

    if matchesMask == None:
        loss = np.inf
    else:
        # map src to dst
        im_image_dst = cv.warpPerspective(unreg_image, M, (w, h))
        mask = np.ones(unreg_image.shape, dtype=np.uint8)
        mask = cv.warpPerspective(mask, M, (w, h))

        threshold = 0.5
        precentage = mask.sum() / (mask.shape[0] * mask.shape[1] * 3)

        det = np.linalg.det(M)
        if 0.5 < det < 2 and precentage > threshold:
            # image_numpy = cv.cvtColor(unreg_image, cv.COLOR_BGR2RGB)
            unreg_image = cv.normalize(unreg_image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            # im_image_dst = cv.cvtColor(im_image_dst, cv.COLOR_BGR2RGB)
            im_image_dst = cv.normalize(im_image_dst, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
            # rgb_numpy = cv.cvtColor(reg_image, cv.COLOR_BGR2RGB)
            reg_image = cv.normalize(reg_image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

            rgb_for_loss = torch.from_numpy(np.moveaxis(reg_image * mask, -1, 0)).to(dtype=float)
            image_for_loss = torch.from_numpy(np.moveaxis(im_image_dst, -1, 0)).to(dtype=float)
            loss = rgb_loss(image_for_loss.unsqueeze(0), rgb_for_loss.unsqueeze(0)) / mask.sum()

            def rotation_matrix_to_angle(rotation_matrix):
                theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                return theta

            # loss = np.abs(rotation_matrix_to_angle(M[:2, :2]))

            cv.imwrite(
                f"/home/leo/nerfstudio_reg/nerfstudio/check/homography_unreg_{unreg_frame}_reg_{reg_frame}_{loss:.7f}_det_{det:.4f}.png",
                np.concatenate((unreg_image, im_image_dst, reg_image), axis=1))
        else:
            loss = np.inf

    return loss, M, src_pts, dst_pts

def get_r_t(unreg_image, reg_image, fl_x, fl_y, cx, cy, src_pts, dst_pts):

    # sift = cv.SIFT_create()
    # # print(unreg_image.shape)
    # kp_image, des_image = sift.detectAndCompute(unreg_image, None)
    # kp_rgb, des_rgb = sift.detectAndCompute(reg_image, None)
    #
    # bf = cv.BFMatcher(crossCheck=True)
    # matches = bf.match(des_image, des_rgb)
    # matches = sorted(matches, key=lambda x: x.distance)
    #
    # good_matches = matches[:50]  # You can adjust the number of good matches
    #
    # # Extract matched keypoints
    # src_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp_rgb[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate essential matrix and recover pose
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = fl_x
    camera_matrix[1, 1] = fl_y
    camera_matrix[0, 2] = cx
    camera_matrix[1, 2] = cy

    fl_avg = (fl_x + fl_y) / 2
    essential_matrix, _ = cv.findEssentialMat(src_pts, dst_pts, camera_matrix, cv.RANSAC, 0.999, 3.0 / fl_avg)
    good, rotation, translation, _ = cv.recoverPose(essential_matrix, src_pts, dst_pts, camera_matrix)
    print(good)
    return rotation, translation



def main(reg_tansforms_path, unreg_tansforms_path, dowanscale_factor):

    dowanscale_factor_int = int(dowanscale_factor)

    with open(os.path.join(reg_tansforms_path), 'r') as f:
        reg_tansforms = json.load(f)

    with open(os.path.join(unreg_tansforms_path), 'r') as f:
        unreg_tansforms = json.load(f)

    reg_tansforms_base = Path(reg_tansforms_path).parent
    unreg_tansforms_base = Path(unreg_tansforms_path).parent

    fl_x = unreg_tansforms["fl_x"] * (1.0/dowanscale_factor_int)
    fl_y = unreg_tansforms["fl_y"] * (1.0/dowanscale_factor_int)
    cx = unreg_tansforms["cx"] * (1.0/dowanscale_factor_int)
    cy = unreg_tansforms["cy"] * (1.0/dowanscale_factor_int)

    best_transform_loss = torch.inf
    for i, unreg_frame in enumerate(unreg_tansforms["frames"]):
        for j , reg_frame in enumerate(reg_tansforms["frames"]):
            # if i!=0 or j!=13:
            #     continue
            unreg_frame_path = unreg_tansforms_base / unreg_frame["file_path"]
            reg_frame_path = reg_tansforms_base / reg_frame["file_path"]
            unreg_mask_path = unreg_tansforms_base / unreg_frame["mask_path"]
            reg_mask_path = reg_tansforms_base / reg_frame["mask_path"]

            if dowanscale_factor != 1:
                unreg_frame_path = str(Path(str(Path(unreg_frame_path).parent) + ("_"+dowanscale_factor)) / Path(unreg_frame_path).name)
                reg_frame_path = str(Path(str(Path(reg_frame_path).parent) + ("_"+dowanscale_factor)) / Path(reg_frame_path).name)

            unreg_image = cv.imread(unreg_frame_path)
            reg_image = cv.imread(reg_frame_path)
            loss, H, src_pts, dst_pts = get_homograph_loss(unreg_image, reg_image, i, j)
            if loss < best_transform_loss:
                best_H = H
                best_transform_loss = loss
                best_transform = (i, j)
                best_unreg_image = unreg_image
                best_reg_image = reg_image
                best_src_pts = src_pts
                best_dst_pts = dst_pts
                print(best_transform_loss)

    unreg_tansforms_matrix = torch.tensor(unreg_tansforms["frames"][best_transform[0]]["transform_matrix"])
    reg_tansforms_matrix = torch.tensor(reg_tansforms["frames"][best_transform[1]]["transform_matrix"])

    print(unreg_tansforms_matrix)

    essential_rot, essential_trans = get_r_t(best_unreg_image, best_reg_image, fl_x, fl_y, cx, cy, best_src_pts, best_dst_pts)
    essential_transform = torch.eye(4)
    essential_transform[:3, :3] = torch.from_numpy(essential_rot)
    # essential_transform[:3, :3] = torch.eye(3)
    # essential_transform[:3, -1] = torch.from_numpy(essential_trans.T) * unreg_tansforms["scale"]

    # # Convert from COLMAP's camera coordinate system to ours
    # essential_transform[0:3, 1:3] *= -1
    # essential_transform = essential_transform[np.array([1, 0, 2, 3]), :]
    # essential_transform[2, :] *= -1

    # essential_transform[:3, :3] = torch.eye(3)


    # a_to_b = torch.eye(4)
    # a_to_b[:3,-1] = unreg_tansforms_matrix[:3, :3].matmul(essential_transform[:3,-1])
    #
    # origin = pose_utils.multiply(unreg_tansforms_matrix, pose_utils.inverse(unreg_tansforms_matrix))
    # origin_rotated = origin[:3, :3].matmul(essential_transform[:3, :3])
    # a_to_b[:3, :3] = origin_rotated
    # print(a_to_b)
    # print(essential_transform[:3,-1])
    # print(unreg_tansforms_matrix[:3,-1])

    # origin = pose_utils.multiply(unreg_tansforms_matrix, pose_utils.inverse(unreg_tansforms_matrix))
    # a_to_b = pose_utils.multiply(origin, essential_transform)
    # print(origin)
    # a_to_b[:3, :3] = torch.eye(3)

    # origin = pose_utils.multiply(unreg_tansforms_matrix, pose_utils.inverse(unreg_tansforms_matrix))
    # origin_rotated = pose_utils.multiply(essential_transform, origin)


    # # Worked:
    # a_rotated = pose_utils.multiply(essential_transform, pose_utils.inverse(unreg_tansforms_matrix))
    # a_to_b = pose_utils.multiply(reg_tansforms_matrix, a_rotated)

    # origin = pose_utils.multiply(pose_utils.inverse(unreg_tansforms_matrix), unreg_tansforms_matrix)
    # a_rotated = torch.eye(4)
    # a_rotated[:3, :3] = essential_transform[:3, :3].matmul(origin[:3, :3])
    # a_to_b = pose_utils.multiply(origin, essential_transform)
    # a_to_b = pose_utils.multiply(a_rotated, unreg_tansforms_matrix)
    # a_to_b[:3, -1] = unreg_tansforms_matrix[:3, -1] + unreg_tansforms_matrix[:3, :3].matmul(essential_transform[:3, -1])
    # a_to_b = pose_utils.multiply(reg_tansforms_matrix, pose_utils.inverse(a_to_b))
    #
    # a_to_b = pose_utils.multiply(unreg_tansforms_matrix, essential_transform)
    # a_to_b = pose_utils.multiply(reg_tansforms_matrix, pose_utils.inverse(a_to_b))
    # a_to_b = pose_utils.multiply(a_to_b, essential_transform)

    # a_rotated = pose_utils.multiply(unreg_tansforms_matrix, essential_transform)
    # a_to_b = pose_utils.multiply(a_rotated, pose_utils.inverse(unreg_tansforms_matrix))

    a_to_b = pose_utils.multiply(unreg_tansforms_matrix, essential_transform)
    a_to_b = pose_utils.multiply(a_to_b, pose_utils.inverse(unreg_tansforms_matrix))

    print(a_to_b)

    translation = a_to_b[:3, -1].T
    r = R.from_matrix(a_to_b[:3, :3])
    rotation = r.as_euler('xyz')
    dof = np.concatenate((translation, rotation), axis=None)
    print(dof)
    print(exp_map_SO3xR3(torch.from_numpy(dof).unsqueeze(0)))
    best_t0 = {"t0": dof.tolist(), "t0_matrix": a_to_b.tolist()}

    with open(unreg_tansforms_base / "best_t0.json", "w") as outfile:
        json.dump(best_t0, outfile, indent=2)




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_to0_list.py <reg_tansforms.json> <unreg_tansforms.json> <dowanscale_factor>")
    else:
        reg_tansforms_path = sys.argv[1]
        unreg_tansforms_path = sys.argv[2]
        dowanscale_factor = sys.argv[3]
        main(reg_tansforms_path, unreg_tansforms_path, dowanscale_factor)