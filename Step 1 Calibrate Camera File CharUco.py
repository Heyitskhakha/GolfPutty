import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os

image_folder = "C:/Users/skha/PycharmProjects/GolfPutty/Calibration_Files_CharUco/*.png"  # folder containing calibration images

# Create ChArUco board
squares_x = 5     # number of squares along X
squares_y = 7     # number of squares along Y

square_length_mm = .00335  # size of a chessboard square in millimeters
marker_length_mm = .0024  # size of an ArUco marker inside a square in millimeters


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
board = aruco.CharucoBoard(
    size=(squares_x, squares_y),
    squareLength=square_length_mm,
    markerLength=marker_length_mm,
    dictionary=aruco_dict
)

# ---------

all_corners = []
all_ids = []
image_size = None

# -------- Load images and detect corners --------
for fname in glob.glob(image_folder):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]  # width, height

    # detect aruco markers
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is not None and len(ids) > 0:
        # interpolate charuco corners
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        #img_drawn = aruco.drawDetectedCornersCharuco(img.copy(), charuco_corners, charuco_ids)
        #cv2.imshow('Charuco Detection', img_drawn)
        #cv2.waitKey(1000)

        # Only use images with at least 4 corners
        if retval is not None and charuco_corners is not None and charuco_ids is not None:
            if len(charuco_corners) >= 2:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
            else:
                print(f"{fname} skipped: only {len(charuco_corners)} corners detected")
# -------- Calibrate camera --------
if len(all_corners) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("Calibration successful!")
    print("RMS reprojection error:", ret)  # <--- This is the RMS error
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    os.makedirs("calibration_results", exist_ok=True)
    np.save("calibration_results/camera_matrix.npy", camera_matrix)
    np.save("calibration_results/dist_coeffs.npy", dist_coeffs)
    print("Saved calibration matrices to calibration_results/")
else:
    print("No ChArUco corners detected. Check images or board parameters.")
