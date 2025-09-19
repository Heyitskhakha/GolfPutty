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
# Replace CHARUCOBOARD with checkerboard parameters

CHECKERBOARD_SQUARE_SIZE = 30.0  # size of each square in mm
picture_range=15
RDK= Robolink()

Poses = [RDK.Item(f'Target {i}') for i in range(1, picture_range)]
for i in range(1, picture_range):
    globals()[f"Pose_{i}"] = RDK.Item(f"Target {i}")

camera= RDK.Item('My Camera')
robot = RDK.Item("Doosan Robotics H2515")
robot.setSpeed(1000)  # mm/s
robot.setAcceleration(1000)  # mm/s²

if not camera.Valid():
    print('CAMERA NOT FOUND')
    exit()

calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"


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

# Get the images
images_dir="C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/"
# Get the images
image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
# Read the image as grayscale
for i, image_path in enumerate(image_files):
    # Read the image as grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    gray = img
    frame_size = gray.shape[::-1]
    if img is None:
        print(f"Failed to read {image_path}")
        continue
    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)
    if ret == True:
        print("detected")
        rcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(rcorners)
        obj_points.append(objp)

        # Draw and display the corners
        cv.drawChessboardCorners(img, PATTERN, rcorners, ret)
        cv.imshow('Original image', img)


# Read the image as grayscale
'''
for i, pose in enumerate(Poses):
    #robot.MoveJ(pose)
    #image_path = os.path.join(images_dir, f"{i + 1}.png")
    #print(f"Capturing image to: {image_path}")
    #success = RDK.Cam2D_Snapshot(image_path, camera)
    img = cv.imread(images_dir)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Original image', gray)
    cv.waitKey(10)
    frame_size = gray.shape[::-1]
    ret, corners = cv.findChessboardCorners(gray, PATTERN, None)
    print("ret:",ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("detected")
        rcorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(rcorners)
        obj_points.append(objp)

        # Draw and display the corners
        cv.drawChessboardCorners(img, PATTERN, rcorners, ret)
        cv.imshow('Original image', img)




cv.destroyAllWindows()
'''
#----------------------------------------------
# Get the calibrated camera parameters
rms_err, calib_mtx, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, frame_size, None, None)
print('Overall RMS re-projection error: %0.3f' % rms_err)

#----------------------------------------------
# Save the parameters
file = getSaveFileName(strfile='RoboDK-Camera-Settings.yaml', defaultextension='.yaml', filetypes=[('YAML files', '.yaml')])
cv_file = cv.FileStorage(file, cv.FILE_STORAGE_WRITE)
if not cv_file.isOpened():
    raise Exception('Failed to save calibration file')
cv_file.write("camera_matrix", calib_mtx)
cv_file.write("dist_coeff", dist_coeffs)
cv_file.write("camera_size", frame_size)
cv_file.release()