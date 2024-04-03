import cv2
import argparse


def merge_videos(video1_path, video2_path, output_path):
    # Open input videos
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Get properties of input videos
    width1 = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = int(video1.get(cv2.CAP_PROP_FPS))
    total_frames1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

    width2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(video2.get(cv2.CAP_PROP_FPS))
    total_frames2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choose the minimum of the two frame counts
    min_total_frames = min(total_frames1, total_frames2)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (width1 + width2, max(height1, height2)))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # Resize frames to fit side by side
        frame1 = cv2.resize(frame1, (int(width1 * min(height1, height2) / height1), min(height1, height2)))
        frame2 = cv2.resize(frame2, (int(width2 * min(height1, height2) / height2), min(height1, height2)))

        # Combine frames side by side
        combined_frame = cv2.hconcat([frame1, frame2])

        # Write the combined frame to the output video
        out.write(combined_frame)

    # Release video objects and writer
    video1.release()
    video2.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge two videos side by side.')
    parser.add_argument('video1', type=str, help='Path to the first input video file.')
    parser.add_argument('video2', type=str, help='Path to the second input video file.')
    parser.add_argument('output', type=str, help='Path to the output video file.')
    args = parser.parse_args()

    merge_videos(args.video1, args.video2, args.output)
