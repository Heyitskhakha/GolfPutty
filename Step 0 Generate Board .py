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

picture_range=15
RDK= Robolink()

Poses = [RDK.Item(f'Target {i}') for i in range(1, picture_range)]
for i in range(1, picture_range):
    globals()[f"Pose_{i}"] = RDK.Item(f"Target {i}")

camera_item= RDK.Item('My Camera')
robot = RDK.Item("Doosan Robotics H2515")
robot.setSpeed(1000)  # mm/s
robot.setAcceleration(1000)  # mm/s²

if not camera_item.Valid():
    print('CAMERA NOT FOUND')
    exit()

calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"
output_dir="C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles"

SQUARES_X = 10  # number of squares along the X axis
SQUARES_Y =  6 # number of squares along the Y axis
PATTERN = (SQUARES_X-1, SQUARES_Y-1)
SQUARE_LENGTH = 30  # mm, length of one square
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
    bytes_img = camera_item.RDK().Cam2D_Snapshot('', camera_item)
    nparr = np.frombuffer(bytes_img, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
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



    cv.waitKey(1)