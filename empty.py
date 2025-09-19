import cv2
import numpy as np

# --- Camera intrinsics (dummy values, replace with yours from calibration) ---
K = np.array([[1500, 0, 960],    # fx, 0, cx
              [0, 1500, 540],    # 0, fy, cy
              [0,    0,   1]], dtype=np.float64)

distCoeffs = np.zeros(5)  # assume no distortion for now (replace with your calibration)

# --- Object points in chessboard frame (mm) ---
objp = np.array([
    [0,   0, 0],
    [30,  0, 0],
    [60,  0, 0],
    [90,  0, 0],
    [120, 0, 0],
    [0,  30, 0]
], dtype=np.float64)

# --- Image points (pixels) ---
imgpts = np.array([
    [1330.5, 329.5],
    [1330.5, 401.5],
    [1330.5, 473.5],
    [1330.5, 544.5],
    [1330.5, 616.5],
    [1259.5, 329.5]
], dtype=np.float64)

# Reshape for solvePnP
objp = objp.reshape(-1,1,3)
imgpts = imgpts.reshape(-1,1,2)

# --- Solve PnP ---
success, rvec, tvec = cv2.solvePnP(objp, imgpts, K, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

if success:
    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # Convert rvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    print("Rotation Matrix (R):\n", R)

    # Test projecting objp back to image
    reprojected, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs)
    print("\nReprojected points:\n", reprojected.reshape(-1,2))