from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d

# ---------------------------
# 1) Load YOLO model
# ---------------------------
# You should train YOLO on Lego images first (lego.pt is a placeholder).
# YOLO will only give bounding boxes in 2D.
model = YOLO("lego.pt")

# ---------------------------
# 2) Configure Intel RealSense
# ---------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # depth in millimeters
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # color in BGR
pipeline.start(config)

# Align depth to color (so pixels match between both)
align = rs.align(rs.stream.color)

# ---------------------------
# 3) Capture one frame (for now)
# ---------------------------
frames = pipeline.wait_for_frames()
frames = align.process(frames)               # align depth to color
color_frame = frames.get_color_frame()       # get RGB frame
depth_frame = frames.get_depth_frame()       # get depth frame

# Convert RealSense frames to numpy arrays (OpenCV / numpy format)
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# ---------------------------
# 4) Run YOLO detection on the color image
# ---------------------------
# Results contain bounding boxes, class IDs, and confidence
results = model.predict(source=color_image, conf=0.5, verbose=False)

for r in results:
    for box in r.boxes:
        # Bounding box corners (x1,y1,x2,y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Draw detection box on color image (for visualization)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0,255,0), 2)

        # Extract region-of-interest from depth and color images
        roi_depth = depth_image[y1:y2, x1:x2]
        roi_color = color_image[y1:y2, x1:x2]

cv2.imshow("YOLO Detection", color_image)
cv2.waitKey(0)

# ---------------------------
# 5) Convert ROI into Open3D point cloud
# ---------------------------
# Get camera intrinsics (fx, fy, cx, cy) from RealSense
intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    intrinsics.width, intrinsics.height,
    intrinsics.fx, intrinsics.fy,
    intrinsics.ppx, intrinsics.ppy
)

# Convert ROI (numpy images) to Open3D Image format
roi_depth_o3d = o3d.geometry.Image(roi_depth)
roi_color_o3d = o3d.geometry.Image(roi_color)

# Create RGBD image (combines color + depth)
roi_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    roi_color_o3d, roi_depth_o3d,
    depth_scale=1000.0,  # RealSense depth is in millimeters
    depth_trunc=1.0,     # ignore everything beyond 1 meter
    convert_rgb_to_intensity=False
)

# Generate point cloud from RGBD image + camera intrinsics
pcd_roi = o3d.geometry.PointCloud.create_from_rgbd_image(
    roi_rgbd, pinhole_intrinsic
)

# Flip orientation (RealSense and Open3D use different coordinate systems)
pcd_roi.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd_roi])  # visualize point cloud of detected Lego

# ---------------------------
# 6) ICP Registration with CAD model
# ---------------------------
# Load CAD model of Lego (STL, PLY, OBJ, etc.)
# NOTE: You’ll need to export your Lego CAD model.
cad_mesh = o3d.io.read_triangle_mesh("lego.stl")
cad_mesh.compute_vertex_normals()

# Convert CAD mesh into a point cloud by sampling points
pcd_model = cad_mesh.sample_points_poisson_disk(1000)

# Run ICP to align Lego ROI cloud with CAD model
threshold = 0.01  # max distance (meters) ICP will consider
trans_init = np.identity(4)  # initial guess (no transform)

reg_result = o3d.pipelines.registration.registration_icp(
    pcd_roi, pcd_model, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Print the estimated transformation matrix (this is the 6D pose!)
print("ICP Transformation:\n", reg_result.transformation)

# Apply transformation to ROI point cloud (so it aligns with CAD model)
pcd_roi.transform(reg_result.transformation)

# Show alignment result
o3d.visualization.draw_geometries([pcd_model, pcd_roi])

# ---------------------------
# 7) Cleanup
# ---------------------------
pipeline.stop()
cv2.destroyAllWindows()
