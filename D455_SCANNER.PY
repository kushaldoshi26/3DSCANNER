import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2 
import os

# 1. Initialize RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()

# Modified for USB 2.0 bandwidth limits
config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

print("Starting D455 stream...")
profile = pipeline.start(config)

# 2. Get Camera Intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

align = rs.align(rs.stream.color)

# 3. Setup Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="D455 3D Scanner", width=848, height=480)

print("Warming up camera sensors...")
for _ in range(30):
    pipeline.wait_for_frames()

pcd = o3d.geometry.PointCloud()
first_frame = True

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        o3d_depth = o3d.geometry.Image(depth_image)
        o3d_color = o3d.geometry.Image(color_image)
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, 
            o3d_depth, 
            depth_scale=1000.0,  
            depth_trunc=3.0,     
            convert_rgb_to_intensity=False)
            
        temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, pinhole_camera_intrinsic)
            
        pcd.points = temp_pcd.points
        pcd.colors = temp_pcd.colors
        
        if first_frame:
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            vis.add_geometry(pcd)
            
            # --- FIXES THE VIEWPOINT AND POINT SIZE ---
            vis.get_render_option().point_size = 3.0 
            vis.reset_view_point(True)
            # ------------------------------------------
            
            first_frame = False
        else:
            vis.update_geometry(pcd)
            
        vis.poll_events()
        vis.update_renderer()
        
except KeyboardInterrupt:
    print("\nStopping stream...")
    print("Saving 3D point cloud capture...")
    
    # --- SAVES THE FILE ---
    filename = "capture.ply"
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Successfully saved '{filename}' to your project folder!")
    # ----------------------
    
finally:
    pipeline.stop()
    vis.destroy_window()