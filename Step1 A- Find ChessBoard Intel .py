import cv2
import numpy as np
import cv2.aruco as aruco

from robodk.robomath import Mat
from robodk.robolink import *

from robodk.robolink import *
from robodk.robodialogs import *
import cv2 as cv
import numpy as np
import glob
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
picture_range=15


calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"
image_folder = "C:/Users/skha/PycharmProjects/GolfPutty/Calibration_Files_CharUco/*.png"  # folder containing calibration images


SQUARES_X = 10  # number of squares along the X axis
SQUARES_Y =  7 # number of squares along the Y axis
PATTERN = (SQUARES_X-1, SQUARES_Y-1)
SQUARE_LENGTH = 25  # mm, length of one square
# Initialize lists for calibration
all_corners = []  # 2D image points
all_objpoints = []  # 3D object points

frame_size = None
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane.
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)  # object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_LENGTH
while(1):
    # Retrieve the image by socket

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
    img=color
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_size = gray.shape[::-1]
    # If found, add object points, image points (after refining them)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)

    if ret:
        rcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Optional: Limit the number of calibration frames
        if len(img_points) < 20:  # Collect up to 20 calibration frames
            img_points.append(rcorners)
            obj_points.append(objp)

        # Draw and display the corners
        cv.drawChessboardCorners(img, PATTERN, rcorners, ret)
        cv.putText(img, f"Chessboard detected! ({len(img_points)} frames)",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv.putText(img, "No chessboard detected",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow('Original image', img)
    key = cv2.waitKey(1)
    if key == ord('k'):
        files = glob.glob(os.path.join(save_folder, "capture_*.png"))
        n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
        filename = os.path.join(save_folder, f"capture_{n}.png")
        cv2.imwrite(filename, color)
        print(f"Saved ROI image: {filename}")


    cv.waitKey(1)