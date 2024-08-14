import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os

# Initialize variables

rgb_directory = r'D:\Internship\MaskRCNN-Depth\color_data'
depth_directory = r'D:\Internship\MaskRCNN-Depth\depth_data'

# Ensure the directories exist
os.makedirs(rgb_directory, exist_ok=True)
os.makedirs(depth_directory, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an align object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

index = 1
print("Starting video stream... Press 's' to save a frame, 'q' to quit.")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply a colormap to depth image for visualization
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.05), cv.COLORMAP_JET)

        # Stack both images horizontally for a side-by-side view
        images = np.hstack((color_image, depth_colormap))

        # Show the stacked images
        cv.imshow('RealSense', images)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            color_filename = os.path.join(rgb_directory, f"Color_frame_{index}.jpg")
            depth_filename = os.path.join(depth_directory, f"Depth_frame_{index}.png")

            # Save color image
            cv.imwrite(color_filename, color_image)

            # Save depth image (16-bit PNG)
            depth_image_8bit = cv.normalize(depth_image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            cv.imwrite(depth_filename, depth_image_8bit)

            print(f"Frame {index} saved to {color_filename}.")
            index += 1

finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
