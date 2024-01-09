import cv2
import numpy as np
import os
import sys

def genDiag(nR, nC, valUpper, valDiag, valLower):
    slope = nC / nR
    tbl = np.full((nR, nC), valDiag, dtype=float)
    for r in range(nR):
        tbl[r, 0 : int(round(slope * (r - 3), 0))] = valLower
        tbl[r, int(round(slope * (r + 4), 0)) : nC] = valUpper
    return tbl

def merge_images_and_videos(input1, input2, output_path):
    # Function to check if the input is an image or video
    def is_image(file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']

    def is_video(file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in ['.mp4', '.avi', '.mkv', '.mov']

    # Read inputs
    if is_image(input1) and is_image(input2):
        img1 = cv2.imread(input1)
        img2 = cv2.imread(input2)

        if img1.shape != img2.shape:
            raise ValueError("Input images must be of the same size")

        # Create a mask
        w = int(img1.shape[0])
        h = int(img1.shape[1])

        mask = genDiag(w, h, 1, 0, -1)[..., None]
        mask1 = mask == 1
        mask2 = mask == -1
        mask_diag = mask == 0

        # Merge images along the diagonal
        merged_image = np.zeros_like(img1)
        merged_image += img1 * mask1
        merged_image += img2 * mask2
        merged_image += np.uint8([255, 255, 255]) * mask_diag

        cv2.imwrite(output_path, merged_image)

    elif is_video(input1) and is_video(input2):
        cap1 = cv2.VideoCapture(input1)
        cap2 = cv2.VideoCapture(input2)

        if (cap1.get(3), cap1.get(4)) != (cap2.get(3), cap2.get(4)):
            raise ValueError("Input videos must have the same resolution")

        h = int(cap1.get(3))
        w = int(cap1.get(4))

        mask = genDiag(w, h, 1, 0, -1)[..., None]
        mask1 = mask == 1
        mask2 = mask == -1
        mask_diag = mask == 0


        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPEG codec
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap1.get(3)), int(cap1.get(4))))

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Merge frames along the diagonal
            merged_frame = np.zeros_like(frame1)
            merged_frame += frame1 * mask1
            merged_frame += frame2 * mask2
            merged_frame += np.uint8([255, 255, 255]) * mask_diag


            out.write(merged_frame)

        cap1.release()
        cap2.release()
        out.release()

    else:
        raise ValueError("Invalid input types. Please provide either two images or two videos.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input1_path input2_path output_path")
        sys.exit(1)

    input_path1 = sys.argv[1]
    input_path2 = sys.argv[2]
    output_path = sys.argv[3]

    merge_images_and_videos(input_path1, input_path2, output_path)
