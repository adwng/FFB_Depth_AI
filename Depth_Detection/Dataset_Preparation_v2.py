import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os
import open3d as o3d

# Set the output directories
data_directory = r'D:\Internship\MaskRCNN-Depth\Data'
os.makedirs(data_directory, exist_ok=True)

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

        # Visualize for confirmation (optional)
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.5), cv.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        cv.imshow('RealSense', images)
        
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Create point cloud
            points = rs.pointcloud()
            points.map_to(color_frame)
            point_cloud = points.calculate(aligned_depth_frame)

            # Convert to Open3D point cloud
            vtx = np.asanyarray(point_cloud.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vtx)

            # Apply a transformation to invert the point cloud if necessary
            # Example transformation: invert along Z-axis
            transformation_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],  # Invert Z-axis
                [0, 0, 0, 1]
            ])
            pc.transform(transformation_matrix)

            # Save color image and point cloud
            color_filename = os.path.join(data_directory, f"image_{index}.png")
            pcd_filename = os.path.join(data_directory, f"point_cloud_{index}.pcd")

            # Save color image
            cv.imwrite(color_filename, color_image)

            # Save point cloud
            o3d.io.write_point_cloud(pcd_filename, pc)

            print(f"Frame {index} saved to {color_filename} and {pcd_filename}.")
            index += 1

finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
