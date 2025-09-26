import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import os
import glob

# Folder to save images
save_folder = "Calibration_Files_CharUco"
os.makedirs(save_folder, exist_ok=True)
# -----------------------------
# 1) Charuco board parameters
# -----------------------------

# -------- Setup RealSense and YOLO ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


squares_x = 5     # number of squares along X
squares_y = 7     # number of squares along Y

square_length_mm = 35/1000  # size of a chessboard square in millimeters
marker_length_mm = 20/1000  # size of an ArUco marker inside a square in millimeters


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard(
    size=(squares_x, squares_y),
    squareLength=square_length_mm,
    markerLength=marker_length_mm,
    dictionary=aruco_dict
)

# -----------------------------
# 2) Setup RealSense
# -----------------------------
#cap = cv2.VideoCapture(0)  # 0 = default webcam
try:
    while True:
        frames = pipeline.wait_for_frames()
        # color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
        align = rs.align(rs.stream.color)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not color_frame or not depth_frame: continue
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # -----------------------------
        # 3) Convert to grayscale for detection
        # -----------------------------
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # -----------------------------
        # 4) Detect ArUco markers
        # -----------------------------
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)

        # If markers detected, interpolate Charuco corners
        if ids is not None and len(ids) > 0:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

            # Draw Charuco corners
            if ret > 0:  # if at least one corner detected
                aruco.drawDetectedCornersCharuco(color, charuco_corners, charuco_ids)

        # Draw all detected ArUco markers
        aruco.drawDetectedMarkers(color, corners, ids)

        # Show result
        cv2.imshow("Charuco Detection", color)
        key = cv2.waitKey(1)
        if key == ord('k'):
            files = glob.glob(os.path.join(save_folder, "capture_*.png"))
            n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
            filename = os.path.join(save_folder, f"capture_{n}.png")
            cv2.imwrite(filename, color)
            print(f"Saved ROI image: {filename}")


finally:

    cv2.destroyAllWindows()
