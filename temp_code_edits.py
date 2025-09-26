
def return_robot_pose(rvec,tvec):
    Robot_Pose = robot.Pose()
    print('Robot_Pose:', Robot_Pose.Rot33())
    # Object rotation in camera frame
    R_obj_cam, _ = cv2.Rodrigues(rvec)  # convert rotation matrix
    t_obj_cam = tvec.reshape(3, 1)  # converts to a [3,1] array

    # Convert object to robot frame
    R_obj_robot = Robot_Pose.Rot33() @ R_obj_cam
    t_obj_robot = R_cam_robot @ t_obj_cam
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

