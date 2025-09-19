import cv2
from robodk import robolink  # RoboDK API
from robodk import robomath  # Robot toolbox
from robodk import robodialogs  # Dialogs

# Uncomment these lines to automatically install the dependencies using pip
# robolink.import_install('cv2', 'opencv-contrib-python==4.5.*')
# robolink.import_install('numpy')
import cv2 as cv
import numpy as np
import json
from pathlib import Path
from enum import Enum
import cv2.aruco as aruco
from robodk.robolink import *
#--------------------------------------
# This scripts supports chessboard (checkerboard) and ChAruCo board as calibration objects.
# You can add your own implementation (such as dot patterns).
class MarkerTypes(Enum):
    CHESSBOARD = 0
    CHARUCOBOARD = 1
INTRINSIC_BOARD_TYPE = MarkerTypes.CHESSBOARD
INTRINSIC_CHESS_SIZE = (6, 10)  # X/Y
INTRINSIC_SQUARE_SIZE = 30  # mm


# The camera intrinsic parameters can be performed with the same board as the Hand-eye
INTRINSIC_FOLDER = 'Hand-Eye-Data'  # Default folder to load images for the camera calibration, relative to the station folder
HANDEYE_FOLDER = 'Hand-Eye-Data'  # Default folder to load robot poses and images for the hand-eye calibration, relative to the station folder


# Hand-eye calibration board parameters
# You can find this charucoboard here: https://docs.opencv.org/4.x/charucoboard.png


def pose_2_Rt(pose: robomath.Mat):
    """RoboDK pose to OpenCV pose"""
    pose_inv = pose.inv()
    R = np.array(pose_inv.Rot33())
    t = np.array(pose.Pos())
    return R, t


def Rt_2_pose(R, t):
    """OpenCV pose to RoboDK pose"""
    vx, vy, vz = R.tolist()

    cam_pose = robomath.eye(4)
    cam_pose.setPos([0, 0, 0])
    cam_pose.setVX(vx)
    cam_pose.setVY(vy)
    cam_pose.setVZ(vz)

    pose = cam_pose.inv()
    pose.setPos(t.tolist())

    return pose



def find_chessboard(img, mtx, dist, chess_size, squares_edge, refine=True, draw_img=None):
    """
    Detects a chessboard pattern in an image.
    """

    pattern = np.subtract(chess_size, (1, 1))  # number of corners

    # Prefer grayscale images
    _img = img
    if len(img.shape) > 2:
        _img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Find the chessboard's corners
    success, corners = cv.findChessboardCorners(_img, pattern)
    if not success:
        raise Exception("No chessboard found")

    # Refine
    if refine:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        search_size = (11, 11)
        zero_size = (-1, -1)
        corners = cv.cornerSubPix(_img, corners, search_size, zero_size, criteria)

    if draw_img is not None:
        cv.drawChessboardCorners(draw_img, pattern, corners, success)

    # Find the camera pose. Only available with the camera matrix!
    if mtx is None or dist is None:
        return corners, None, None

    cb_corners = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    cb_corners[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * squares_edge
    success, rvec, tvec = cv.solvePnP(cb_corners, corners, mtx, dist)
    if not success:
        raise Exception("No chessboard found")

    R_target2cam = cv.Rodrigues(rvec)[0]
    t_target2cam = tvec

    return corners, R_target2cam, t_target2cam
def calibrate_static(chessboard_images, board_type: MarkerTypes, chess_size, squares_edge: float, min_detect: int = -1, show_images=False):
    """
    Calibrate a camera with a chessboard or charucoboard pattern.
    """
    # Chessboard parameters
    pattern = np.subtract(chess_size, (1, 1))  # number of corners
    img_size = None

    # Find the chessboard corners
    img_corners = []

    if show_images:
        WDW_NAME = 'Chessboard'
        MAX_W, MAX_H = 1920, 1080
        cv.namedWindow(WDW_NAME, cv.WINDOW_NORMAL)

    for file, img in chessboard_images.items():
        # Ensure the image size is consistent
        if img_size is None:
            img_size = img.shape[:2]
        else:
            if img.shape[:2] != img_size:
                raise Exception('Camera resolution is not consistent across images!')

        # Find the chessboard corners
        draw_img = None
        if show_images:
            draw_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        try:
            if board_type == MarkerTypes.CHESSBOARD:
                corners, _, _ = find_chessboard(img, mtx=None, dist=None, chess_size=chess_size, squares_edge=squares_edge, draw_img=draw_img)
        except:
            print(f'Unable to find chessboard in {file}!')
            continue

        if show_images:
            cv.imshow(WDW_NAME, draw_img)
            cv.resizeWindow(WDW_NAME, MAX_W, MAX_H)
            cv.waitKey(500)
        img_corners.append(corners)

        # Check if we processed enough images
        if min_detect > 0 and len(img_corners) >= min_detect:
            break

    if show_images:
        cv.destroyAllWindows()

    if len(img_corners) < 3 or min_detect > 0 and len(img_corners) < min_detect:
        raise Exception('Not enough detections!')

    # Calibrate the camera
    # Create a flat chessboard representation of the corners
    cb_corners = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    cb_corners[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * squares_edge

    h, w = img_size
    rms_err, mtx, dist, rot_vecs, trans_vecs = cv.calibrateCamera([cb_corners for i in range(len(img_corners))], img_corners, (w, h), None, None)
    return mtx, dist, (w, h),rms_err

def calibrate_handeye(robot_poses, chessboard_images, camera_matrix, camera_distortion, board_type: MarkerTypes, chess_size, squares_edge: float, markers_edge: float, show_images=False):
    """
    Calibrate a camera mounted on a robot arm using a list of robot poses and a list of images for each pose.
    The robot pose should be at the flange (remove .PoseTool) unless you have a calibrated tool.
    """
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b

    # Rotation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame (bTg).
    # This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from gripper frame to robot base frame.
    R_gripper2base_list = []

    # Translation part extracted from the homogeneous matrix that transforms a point expressed in the gripper frame to the robot base frame (bTg).
    # This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from gripper frame to robot base frame.
    t_gripper2base_list = []

    # Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame (cTt).
    # This is a vector (vector<Mat>) that contains the rotation, (3x3) rotation matrices or (3x1) rotation vectors, for all the transformations from calibration target frame to camera frame.
    R_target2cam_list = []

    # Rotation part extracted from the homogeneous matrix that transforms a point expressed in the target frame to the camera frame (cTt).
    # This is a vector (vector<Mat>) that contains the (3x1) translation vectors for all the transformations from calibration target frame to camera frame.
    t_target2cam_list = []

    if show_images:
        WDW_NAME = 'Charucoboard'
        MAX_W, MAX_H = 640, 480
        cv.namedWindow(WDW_NAME, cv.WINDOW_NORMAL)

    for i in chessboard_images.keys():
        robot_pose = robot_poses[i]
        image = chessboard_images[i]
        draw_img = None
        if show_images:
            draw_img = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        try:
            if board_type == MarkerTypes.CHESSBOARD:
                _, R_target2cam, t_target2cam = find_chessboard(image, camera_matrix, camera_distortion, chess_size, squares_edge, draw_img=draw_img)
            else:
                _, R_target2cam, t_target2cam = find_charucoboard(image, camera_matrix, camera_distortion, chess_size, squares_edge, markers_edge, draw_img=draw_img)
        except:
            print(f'Unable to find chessboard in {i}!')
            continue

        if show_images:
            cv.imshow(WDW_NAME, draw_img)
            cv.resizeWindow(WDW_NAME, MAX_W, MAX_H)
            cv.waitKey(500)

        R_target2cam_list.append(R_target2cam)
        t_target2cam_list.append(t_target2cam)

        R_gripper2base, t_gripper2base = pose_2_Rt(robot_pose)
        R_gripper2base_list.append(R_gripper2base)
        t_gripper2base_list.append(t_gripper2base)

    if show_images:
        cv.destroyAllWindows()

    R_cam2gripper, t_cam2gripper = cv.calibrateHandEye(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list)
    return Rt_2_pose(R_cam2gripper, t_cam2gripper)

def runmain():
    #------------------------------------------------------
    # Calibrate the camera intrinsic parameters
    # 1. Print a chessboard, measure it using a caliper
    # 2. Mount the camera statically, take a series of images of the chessboard at different distance, orientation, offset, etc.
    # 3. Calibrate the camera using the images (can be done offline)
    #
    #
    # Calibrate the camera location (hand-eye)
    # 4. Create a robot program in RoboDK that moves the robot around a static chessboard at different distance, orientation, offset, etc.
    # 5. At each position, record the robot pose (robot.Pose(), or robot.Joints() even) and take a screenshot with the camera
    # 6. Use the robot poses and the images to calibrate the camera location
    #
    #
    # Good to know
    # - You can retrieve the camera image live with OpenCV using cv.VideoCapture(0, cv.CAP_DSHOW)
    # - You can load/save images with OpenCV using cv.imread(filename) and cv.imwrite(filename, img)
    # - You can save your calibrated camera parameters with JSON, i.e. print(json.dumps({"mtx":mtx, "dist":dist}))
    #
    #------------------------------------------------------

    RDK = robolink.Robolink()

    #------------------------------------------------------
    # Calibrate a camera using local images of chessboards, retrieves the camera intrinsic parameters
    # Get the input folder
    #intrinsic_folder = Path(RDK.getParam(robolink.PATH_OPENSTATION)) / INTRINSIC_FOLDER
    #intrinsic_folder= 'C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/'
    intrinsic_folder = Path("C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/")

    # Retrieve the images
    image_files = sorted(intrinsic_folder.glob("*.png"))

    if not image_files:
        raise FileNotFoundError(f"No .png images found in {intrinsic_folder}")

    # Debug: check each file
    for img_path in image_files:
        print(img_path, "exists:", img_path.exists())

    # Read all images in grayscale
    images = {}
    for image_file in image_files:
        img = cv.imread(str(image_file), cv.IMREAD_GRAYSCALE)  # ✅ use str()
        if img is None:
            print(f"[ERROR] Could not load image: {image_file}")
            continue
        images[int(image_file.stem)] = img

    print(f"Loaded {len(images)} images successfully.")
    # Perform the image calibration
    mtx, dist, size,rms = calibrate_static(images, INTRINSIC_BOARD_TYPE, INTRINSIC_CHESS_SIZE, 30, min_detect=-1)
    print(f'Camera matrix:\n{mtx}\n')
    print(f'Distortion coefficient:\n{dist}\n')
    print(f'Camera resolution:\n{size}\n')
    print(f"Reprojection error: {rms:.2f} pixels")




if __name__ == '__main__':
    runmain()