import cv2
import cv2.aruco as aruco
import numpy as np
from robodk.robolink import *
from robodk.robomath import *  # RoboDK Math functions
from robodk import robomath as rm
from robodk.robodialogs import *
# Use your board's dictionary
import pyrealsense2 as rs
import robodk as rdk

from enum import Enum

import numpy as np
import cv2

from math import atan2, sqrt, degrees
def camera_instrinic_parameters(calibration_file_path):

    cv_file = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)

    CAMERA_MTX = cv_file.getNode("camera_matrix").mat()
    DIST_COEFFS = cv_file.getNode("dist_coeff").mat()
    return CAMERA_MTX , DIST_COEFFS

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
#pipeline.start(config)


pattern =[]

class MarkerTypes(Enum):
    CHESSBOARD = 0
    CHARUCOBOARD = 1
INTRINSIC_BOARD_TYPE = MarkerTypes.CHESSBOARD
INTRINSIC_CHESS_SIZE = (10, 6)  # X/Y
INTRINSIC_SQUARE_SIZE = 30  # mm
PATTERN = (INTRINSIC_CHESS_SIZE[0]-1, INTRINSIC_CHESS_SIZE[1]-1)

obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane.objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)  # object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)  # object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * INTRINSIC_SQUARE_SIZE

calibration_file = r"C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/RoboDK-Camera-Settings.yaml"

RDK=Robolink()
camera= RDK.Item('My Camera')
cam_item = RDK.Item('My Camera')

robot = RDK.Item("Doosan Robotics H2515")
Tool_1=RDK.Item('Tool 1')
Tool_2=RDK.Item('Tool 2')
robot.setPoseTool(Tool_1)

Starting_pose= RDK.Item('Home')
robot.MoveJ(Starting_pose)
mtx , dist = camera_instrinic_parameters(calibration_file)

def return_robot_pose(rvec, tvec):
    R_cam_robot = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    Robot_Pose = robot.Pose()
    print('Robot_Pose:', Robot_Pose.Rot33())
    # Object rotation in camera frame
    R_obj_cam, _ = cv2.Rodrigues(rvec)  # convert rotation matrix
    t_obj_cam = tvec.reshape(3, 1)  # converts to a [3,1] array

    # Convert object to robot frame
    R_obj_robot = Robot_Pose.Rot33() @ R_obj_cam
    t_obj_robot = R_cam_robot @ t_obj_cam #using R_cam_robot seems have better accuracy than using the Robot_pose.Rot33()
    print('R_obj_robot:', R_obj_robot)
    # Homogeneous transform of object in robot coordinates
    T_obj_robot = np.eye(4)
    T_obj_robot[:3, :3] = R_obj_robot
    T_obj_robot[:3, 3] = t_obj_robot.flatten()
    # Only translate — keep the current rotation
    T_obj_in_robot_base = np.eye(4)
    T_obj_in_robot_base[:3, :3] = R_obj_robot  # robot's current rotation
    T_obj_in_robot_base[:3, 3] = Robot_Pose.Pos() + t_obj_robot.flatten() + [2, 0, 0]  # add translation only

    # Convert to RoboDK Mat
    Pose_obj_in_robot_mat = Mat(T_obj_in_robot_base.tolist())

    return Pose_obj_in_robot_mat

    # print('Marker_pose:',Marker_pose.Pose())

def find_chessboard(img, mtx, dist, chess_size, squares_edge, refine=True, draw_img=None):
    """
    Detects a chessboard pattern in an image.
    """
    pattern = np.subtract(chess_size, (1, 1))  # number of corners
    # Prefer grayscale images
    # Find the chessboard's corners
    success, corners = cv2.findChessboardCorners(img, pattern)
    print('success:', success)
    if not success:
        raise Exception("No chessboard found")
    # Refine
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        search_size = (11, 11)
        zero_size = (-1, -1)
        corners = cv2.cornerSubPix(img, corners, search_size, zero_size, criteria)

    if draw_img is not None:
        cv2.drawChessboardCorners(draw_img, pattern, corners, success)

    # Find the camera pose. Only available with the camera matrix!
    if mtx is None or dist is None:
        return corners, None, None

    cb_corners = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    cb_corners[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * squares_edge
    success, rvec, tvec = cv2.solvePnP(cb_corners, corners, mtx, dist)

    if not success:
        raise Exception("No chessboard found")
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
    rvec[:] = np.round(rvec, 3)
    #print("rvec:",rvec)
    #print("tvec:",tvec)
    return corners, rvec, tvec

def grab_camera_view(camera_type):
    if(camera_type=="intel"):
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert to numpy
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    elif(camera_type=="robodk"):
        # RoboDK CAMERA
        bytes_img = camera.RDK().Cam2D_Snapshot('', camera)
        nparr = np.frombuffer(bytes_img, np.uint8)
        Img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        color_image = Img
        gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    else:
        print('no camera selected')
        exit()
    return gray,color_image
while True:
    gray,color_image=grab_camera_view('robodk')
    corners,rvec,tvec=find_chessboard(gray, mtx, dist, INTRINSIC_CHESS_SIZE, 30, refine=True, draw_img=None)
    # Finding ChesseBoard for Testing
    '''
    ret, corners = cv2.findChessboardCorners(gray, pattern)
    if ret:
        rcorners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Optional: Limit the number of calibration frames
        if len(img_points) < 20:  # Collect up to 20 calibration frames
            img_points.append(rcorners)
            obj_points.append(objp)

        # Draw and display the corners
        #cv2.drawChessboardCorners(frame, pattern, rcorners, ret)
        cv2.putText(color_image, f"Chessboard detected! ({len(img_points)} frames)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(color_image, "No chessboard detected",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.imshow('Original image', frame)
    #cv2.waitKey(0)
    corners,rvec,tvec=find_chessboard(gray, mtx, dist, INTRINSIC_CHESS_SIZE, 30, refine=True, draw_img=None)
    rvec, tvec = cv2.solvePnPRefineLM(objp, corners, CAMERA_MTX, DIST_COEFFS, rvec, tvec)
    imgpts, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    '''

    #print("rvec:", rvec.ravel())
    Robot_Pose = return_robot_pose(rvec, tvec)
    Marker_pose = RDK.Item('Marker')
    Marker_pose.setPose(Robot_Pose)
    print('Pose_obj_in_robot_mat:', Robot_Pose)
    robot.MoveL(Marker_pose)


    cv2.imshow("Marker Pose", color_image)
    cv2.waitKey(0)
    exit()




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

