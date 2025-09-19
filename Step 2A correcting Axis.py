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


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
#pipeline.start(config)


def align_z_up(rvec):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Desired Z axis in world/camera coordinates
    z_up = np.array([0, 0, 1], dtype=float)

    # Keep the current X axis as close as possible to original
    x_axis = R[:, 0]
    # Recompute Y as cross product to keep orthogonal
    y_axis = np.cross(z_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Recompute X as orthogonal to both Y and Z
    x_axis = np.cross(y_axis, z_up)
    x_axis /= np.linalg.norm(x_axis)

    # Build the new rotation matrix
    R_new = np.column_stack((x_axis, y_axis, z_up))

    # Convert back to rotation vector
    rvec_new, _ = cv2.Rodrigues(R_new)
    return rvec_new, R_new
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

file = calibration_file
cv_file = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)

CAMERA_MTX = cv_file.getNode("camera_matrix").mat()
DIST_COEFFS = cv_file.getNode("dist_coeff").mat()
mtx = CAMERA_MTX
dist = DIST_COEFFS

RDK=Robolink()
camera= RDK.Item('My Camera')
cam_item = RDK.Item('My Camera')
print(cam_item.Pose())
robot = RDK.Item("Doosan Robotics H2515")


print('mtx:',mtx)
Starting_pose= RDK.Item('Home')
robot.MoveJ(Starting_pose)


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
    print("rvec:",rvec)
    print("tvec:",tvec)
    return corners, rvec, tvec

while True:
    '''
    frames = pipeline.wait_for_frames()
    #depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    # Convert to numpy
    #depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    '''
    #RoboDK CAMERA
    bytes_img = camera.RDK().Cam2D_Snapshot('', camera)
    nparr = np.frombuffer(bytes_img, np.uint8)
    Img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    color_image=Img
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    pattern = np.subtract(INTRINSIC_CHESS_SIZE, (1, 1))  # number of corners
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
    axis_length = 50
    #cv2.drawFrameAxes(color_image, mtx, dist, rvec, tvec, axis_length)
    # Draw projected points on the image
    proj_img = color_image.copy()
    for pt in imgpts:
        x, y = pt.ravel()
        cv2.circle(color_image, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
    #print("rvec:", rvec.ravel())
    #print("first objp points (units mm):")

    #for i in range(6):
    #    print(i, corners[i].ravel())
    #    pass
    if rcorners is not None:
        marker_length = INTRINSIC_SQUARE_SIZE

        # Camera to Robot axis conversion
        R_cam_robot = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])
        Robot_Pose = robot.Pose()
        print('Robot_Pose:',Robot_Pose.Rot33())
        # Object rotation in camera frame
        R_obj_cam, _ = cv2.Rodrigues(rvec) #convert rotation matrix
        t_obj_cam = tvec.reshape(3, 1)#converts to a [3,1] array

        # Convert object to robot frame
        R_obj_robot = Robot_Pose.Rot33() @ R_obj_cam
        t_obj_robot = R_cam_robot@ t_obj_cam
        print('R_obj_robot:',R_obj_robot)
        # Homogeneous transform of object in robot coordinates
        T_obj_robot = np.eye(4)
        T_obj_robot[:3, :3] = R_obj_robot
        T_obj_robot[:3, 3] = t_obj_robot.flatten()
        # Only translate — keep the current rotation
        T_obj_in_robot_base = np.eye(4)
        T_obj_in_robot_base[:3, :3] = R_obj_robot # robot's current rotation
        T_obj_in_robot_base[:3, 3] = Robot_Pose.Pos() + t_obj_robot.flatten() +[2,0,0] # add translation only

        # Convert to RoboDK Mat
        Pose_obj_in_robot_mat = Mat(T_obj_in_robot_base.tolist())
        # Move Marker
        Marker_pose = RDK.Item('Marker')
        Marker_pose.setPose(Pose_obj_in_robot_mat)
        print('Pose_obj_in_robot_mat:', Pose_obj_in_robot_mat)

        #print('Marker_pose:',Marker_pose.Pose())

        robot.MoveL(Marker_pose)
        #robot.MoveL(Marker_pose)
        #time.sleep(1)
        # Draw axis at this corner (length = marker_length/2 for visibility)


        # (Optional) Mark the corner in blue for clarity
        #img_proj, _ = cv2.projectPoints(corner3D, rvec, tvec, mtx, dist)
        #p_proj = tuple(np.round(img_proj[0][0]).astype(int))
        #cv2.circle(frame, p_proj, 6, (255, 0, 0), -1)

        cv2.imshow("Marker Pose", color_image)
        cv2.waitKey(0)
        exit()




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

