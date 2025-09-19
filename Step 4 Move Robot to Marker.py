import cv2
import cv2.aruco as aruco
import numpy as np
from robodk.robolink import *
from robodk.robomath import *  # RoboDK Math functions
from robodk.robodialogs import *
# Use your board's dictionary



import numpy as np
import cv2


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



board = cv2.aruco.CharucoBoard(
    size=(5,7),
    squareLength=0.035,
    markerLength=0.021,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
)
calibration_file = "C:/Users/skha/PycharmProjects/ROBODKPROGRAMS/calibration folder/RoboDK-Camera-Settings.yaml"

file = getOpenFileName(strfile='RoboDK-Camera-Settings.yaml', strtitle='Open calibration settings file..', defaultextension='.yaml', filetypes=[('YAML files', '.yaml')])
cv_file = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)

CAMERA_MTX = cv_file.getNode("camera_matrix").mat()
DIST_COEFFS = cv_file.getNode("dist_coeff").mat()



marker_length=.021
RDK=Robolink()
camera= RDK.Item('My Camera')
cam_item = RDK.Item('My Camera')
print(cam_item.Pose())
robot = RDK.Item("Doosan Robotics H2515")

mtx = CAMERA_MTX
dist = DIST_COEFFS

Starting_pose= RDK.Item('Home')

Marker_pose= RDK.Item('markers')
robot.MoveJ(Starting_pose)
detector = aruco.CharucoDetector(board)
#zero_dist = np.zeros_like(dist)
#dist=zero_dist
while True:
    bytes_img = RDK.Cam2D_Snapshot("", camera)
    nparr = np.frombuffer(bytes_img, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

    if marker_ids is not None:
        aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        # Iterate through all detected markers
        for i, marker_id in enumerate(marker_ids.flatten()):
            if marker_id == 8:  # The marker you care about
                # Estimate pose of the marker

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    marker_corners[i], marker_length, mtx, dist
                )
                rvec, tvec = np.round(rvec, decimals=5),np.round(tvec, decimals=5)
                print("marker_length:", marker_length)
                # Draw axis on that marker
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.02)
                #tvec, rvec = np.round(tvec, 7), np.round(rvec, 7)
                # Display coordinates
                x, y , z = tvec[0][0]*1000

                cv2.putText(
                    frame,
                    f"Marker {marker_id}: X={x:.3f}m Y={y:.3f}m Z={z:.3f}m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                print(f"Marker {marker_id} coordinates: {x:.3f}, {y:.3f}, {z:.3f}")
                obj_points = np.array([
                    [-marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ])
                obj_pts = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * (marker_length / 2)
                img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)

                # Draw reprojected points (should overlap with detected corners)
                for pt in img_pts.squeeze():
                    cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
                imgpts, _ = cv2.projectPoints(obj_points, rvec, tvec, mtx, dist)


                cv2.imshow("Marker Pose", frame)
                cv2.waitKey(5000)
    #cv2.imshow("Marker Pose", frame)


    # Convernt Camera to Base
    #_________________________________________________
    Robot_Pose = robot.Pose()
    R_flange = Robot_Pose.Rot33()
    T_flange = np.array(Robot_Pose.Pos())  # mm

    print('Robot_Pose:',Robot_Pose)
    print('R_flange:', R_flange)
    print('T_flange:', T_flange)
    #T_flange=[0,0,0]
    #-------------------------------------------------
    # --- Camera offset in flange coordinates ---
    cam_offset_translation = np.array([100, 0,50])  # mm from flange origin
    cam_offset_rotation = np.eye(3)  # adjust if tilted
    # --- Flange → Camera transform ---
    Pose_flange_to_cam = np.eye(4)
    Pose_flange_to_cam[:3, :3] = cam_offset_rotation
    Pose_flange_to_cam[:3, 3] = cam_offset_translation

    # --- Base → Flange transform ---
    Pose_base_to_flange = np.eye(4)
    Pose_base_to_flange[:3, :3] = R_flange
    Pose_base_to_flange[:3, 3] = T_flange

    # --- Base → Camera transform ---
    Pose_cam_in_robot = Pose_base_to_flange @ Pose_flange_to_cam
    Pose_cam_in_robot = np.round(Pose_cam_in_robot,5)

    print('Pose_cam_in_robot:',Pose_cam_in_robot)


    #Figure Out Object Position in Camera
    # Convert ArUco to Camera Coordinates
    #-------------------------------------------------
    R_matrix, _ = cv2.Rodrigues(rvec)
    tvec_mm = tvec.flatten()*1000 # meters → mm
    rvec_aligned, R_aligned = align_z_up(rvec)
    Pose_obj_in_cam = np.eye(4)
    # 3. Build object pose in camera frame (4x4 matrix)
    Pose_obj_in_cam[:3, :3] = R_aligned
    # Returns a vector that the object is perpendicular to the camera.
    Pose_obj_in_cam[:3, 3] = tvec_mm

    #-------------------------------------------------
    #Pose_obj_in_robot=Pose_cam_in_robot_rounded@Pose_obj_in_cam_rounded
    Pose_base_to_cam = Pose_base_to_flange @ Pose_flange_to_cam
    Pose_obj_in_robot = Pose_base_to_cam @ Pose_obj_in_cam
    robot_pose_mat = Mat(Pose_obj_in_robot.tolist())
    print("robot_pose_mat:",robot_pose_mat)
    cv2.imshow("Marker Pose", frame)

    robot.setSpeed(1000)  # mm/s
    robot.setAcceleration(1000)  # mm/s²
    # 🔹 Ensure the robot retains the current rotation while setting the new position
    Marker_pose.setPose(robot_pose_mat)
    robot.MoveJ(Marker_pose)
    # Move the robot to the marker pose
    robot.WaitMove()
    print("Pose_base_to_flange:\n", Pose_base_to_flange)
    print("Pose_flange_to_cam:\n", Pose_flange_to_cam)
    print("Pose_obj_in_cam:\n", Pose_obj_in_cam)
    print("Pose_base_to_cam:\n", Pose_base_to_cam)
    print("Pose_obj_in_robot:\n", Pose_obj_in_robot)

    exit()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

