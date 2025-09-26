"""
Candy plane extraction pipeline with debug visualization:
- YOLO detection -> shrink bbox inward by 10%
- sample depth in ROI -> deproject pixels to 3D
- RANSAC plane fit -> SVD refine on inliers
- compute centroid + orientation (rotation) for plane
- visualize inliers/outliers on 2D image
"""
'''
This is the main code and it is currently working with Doosan Robotics Code.

import struct
set_velx(250,500)      # The global task velocity is set to 30(mm/sec) and 20(deg/sec).
set_accx(200,100)   
set_digital_output(1,OFF)
set_digital_output(2,OFF)   
# Read 12 Modbus registers (2 per axis)
set_tcp("OnRobot")    
while(1):

    
        
        
    
     
    x1, sol = get_current_posx() 
    set_modbus_slave(150, 0)
    movej(posj(-93.02, -17.81, -112.39, 0.02, -48.92, 93.49),vel=30,acc=50)
    set_modbus_slave(150, 1)
    while(get_modbus_slave(150)!=2):
        wait(.5)
    registers = [get_modbus_slave(i) for i in range(128, 140)]
    
    # Reconstruct 6 floats
    pose = []
    for i in range(0, 12, 2):
        hi, lo = registers[i], registers[i+1]
        b = struct.pack('>HH', hi, lo)   # Big-endian: hi first
        f = struct.unpack('>f', b)[0]
        pose.append(f)
    
    set_modbus_slave(150, 0)
    # Store as [x, y, z, Rx, Ry, Rz]
    print("Stored pose:", pose)
    new_pose=trans(pose,[0,0,10,0,0,0],DR_TOOL,DR_BASE)
    new_pose[3]=x1[3]
    new_pose[4]=x1[4]
    new_pose[5]=x1[5]
    pose[3]=x1[3]
    pose[4]=x1[4]
    pose[5]=x1[5]
    set_digital_output(2,OFF)
    movel(new_pose)
    movel(pose)
    set_digital_output(1,ON)
    wait(1)
    new_pose=trans(pose,[0,0,-100,0,0,0],DR_TOOL,DR_BASE)
    movel(new_pose,r=20)
    movej(posj(-114.51, -38.01, -94.7, 0.04, -47.21, 72.01),vel=60,acc=60,r=10)    
    set_digital_output(2,ON)
    set_digital_output(1,OFF)
    movej(posj(-93.02, -17.81, -112.39, 0.02, -48.92, 93.49),vel=60,acc=60)
    
    
'''
import time
import math
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import queue
# -------- CONFIG --------
MODEL_WEIGHTS = "C:/Users/skha/PycharmProjects/GolfPutty/lego_training/Red Lego63/weights/best.pt"
CONF_THRESHOLD = 0.60
SHRINK_FACTOR = 0.10
MAX_RANSAC_ITERS = 200
RANSAC_INLIER_THRESH = 0.008   # 8 mm
MIN_POINTS_FOR_PLANE = 50
SAMPLE_STEP = 4

#intel Camera Setup
    # replace with calibrated transform if available
x_robot , y_robot, z_robot, Rx_robot, Ry_robot, Rz_robot= [109.930,815.30,514,87.95,179.12,-85.53]
robot_start_xyz = np.array([x_robot, y_robot, z_robot])
robot_start_rpy = np.array([Rx_robot, Ry_robot, Rz_robot])
robot_start_rvec=np.array([-3.1211,0.1777,0.00050])

robot_start_mat,_=cv2.Rodrigues(robot_start_rvec)
print(robot_start_mat)
camera_to_robot = np.eye(4)
camera_to_robot[:3, :3] = robot_start_mat
camera_to_robot[:3, 3]=robot_start_xyz
import struct
# -------- Utility functions --------

import socket
import time
# ---------- CONFIG ----------
DOOSAN_IP = "192.168.137.100"
DOOSAN_PORT = 502   # default Modbus TCP port
REG_START = 128    # starting register on the robot

from pyModbusTCP.client import ModbusClient
client = ModbusClient(host=DOOSAN_IP, port=DOOSAN_PORT, auto_open=True)


# ---------- HELPER: float -> 2 x 16-bit registers ----------
def float_to_registers(value):
    # pack float as 4 bytes (big endian) then split into two 16-bit registers
    b = struct.pack('>f', value)
    r1 = int.from_bytes(b[0:2], byteorder='big')
    r2 = int.from_bytes(b[2:4], byteorder='big')
    return [r1, r2]






def shrink_bbox(x1, y1, x2, y2, factor):
    w = x2 - x1; h = y2 - y1
    dx = int(w * factor); dy = int(h * factor)
    return x1+dx, y1+dy, x2-dx, y2-dy

def deproject_pixel(u, v, depth_m, intrinsics):
    if depth_m <= 0 or np.isnan(depth_m):
        return None
    x = (u - intrinsics.ppx) * depth_m / intrinsics.fx
    y = (v - intrinsics.ppy) * depth_m / intrinsics.fy
    z = depth_m
    return np.array([x, y, z], dtype=float)

def fit_plane_svd(points):
    centroid = points.mean(axis=0)
    pts_centered = points - centroid
    _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
    normal = vh[-1, :]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid

def ransac_plane(points, iters=200, thresh=0.008):
    best_inliers, best_count, best_plane = None, 0, None
    N = points.shape[0]
    if N < 3: return None, None, None

    for _ in range(iters):
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = points[idx]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-6: continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, p1)
        dist = np.abs(points.dot(n) + d)
        mask = dist < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count, best_inliers, best_plane = count, mask, (n, d)

    if best_plane is None: return None, None, None
    inlier_pts = points[best_inliers]
    if inlier_pts.shape[0] < 3: return None, None, None

    refined_n, centroid = fit_plane_svd(inlier_pts)
    if np.dot(refined_n, best_plane[0]) < 0:
        refined_n = -refined_n
    refined_d = -np.dot(refined_n, centroid)
    final_dist = np.abs(points.dot(refined_n) + refined_d)
    final_mask = final_dist < thresh
    return refined_n, refined_d, final_mask
def return_robot_pose(object_pose,robot_pose):
    rmat_object= object_pose[:3, :3]
    tmat_object = object_pose[:3, 3]
    print('tmat_object:',tmat_object)
    #print('tmat_object:',tmat_object)
    rmat_robot= robot_pose[:3, :3]
    tmat_robot = robot_pose[:3, 3]


    R_cam_robot = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
    ])


    # Object rotation in camera frame
    #R_obj_cam, _ = cv2.Rodrigues(rvec)  # convert rotation matrix
    #t_obj_cam = tvec.reshape(3, 1)  # converts to a [3,1] array

    # Convert object to robot frame
    R_obj_robot = rmat_robot @ rmat_object
    t_obj_robot = R_cam_robot @ tmat_object #using R_cam_robot seems have better accuracy than using the Robot_pose.Rot33()
    #print('R_obj_robot:', R_obj_robot)
    # Homogeneous transform of object in robot coordinates
    T_obj_robot = np.eye(4)
    T_obj_robot[:3, :3] = R_obj_robot
    T_obj_robot[:3, 3] = t_obj_robot.flatten()
    # Only translate — keep the current rotation
    T_obj_in_robot_base = np.eye(4)
    T_obj_in_robot_base[:3, :3] = R_obj_robot  # robot's current rotation
    T_obj_in_robot_base[:3, 3] = tmat_robot + t_obj_robot.flatten() + [-10, -10, 0]  # add translation only

    # Convert to RoboDK Mat
    #print(T_obj_in_robot_base)

    return T_obj_in_robot_base
def compute_pose_from_points_center_only(points):
    """
    Compute a stable 4x4 pose using the centroid of the points.
    Rotation is identity (no rotation).
    """
    centroid = points.mean(axis=0)  # x, y, z in meters
    T = np.eye(4)
    T[:3, 3] = centroid * 1000  # convert to mm if needed
    T[:3, :3] = np.eye(3)       # identity rotation
    return T

def rotation_matrix_to_euler(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1]); y = math.atan2(-R[2,0], sy); z = 0
    return np.array([x,y,z])

import numpy as np

def rotmat_to_euler_zyz(R):
    """
    Convert a 3x3 rotation matrix to ZYZ Euler angles.
    Angles are returned in radians: (alpha, beta, gamma)
    Convention: R = Rz(alpha) * Ry(beta) * Rz(gamma)
    """
    if abs(R[2,2]) < 1.0:  # Normal case
        beta = np.arccos(R[2,2])
        alpha = np.arctan2(R[1,2], R[0,2])
        gamma = np.arctan2(R[2,1], -R[2,0])
    else:
        # Singularity: beta = 0 or pi
        beta = 0 if R[2,2] > 0 else np.pi
        alpha = np.arctan2(R[0,1], R[0,0])
        gamma = 0.0

    return alpha, beta, gamma

def display_thread():
    while True:
        frame = frame_queue.get()  # Wait for a new frame
        if frame is None:          # Sentinel to stop the thread
            break
        cv2.imshow("intel", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break
    cv2.destroyAllWindows()


def wait_for_register_bit(addr, bit_index=0, poll_interval=0.05, timeout=10):
    """
    Wait until a Modbus register's specific bit is 1.

    addr: register address
    bit_index: which bit of the register (0-15)
    poll_interval: seconds between reads
    timeout: max seconds to wait
    """
    start_time = time.time()
    while True:
        regs = client.read_holding_registers(addr, 1)
        if regs is None:
            print("Failed to read register, retrying...")
            time.sleep(poll_interval)
            continue

        reg_val = regs[0]
        if (reg_val >> bit_index) & 1 == 1:
            return True

        if (time.time() - start_time) > timeout:
            print("Timeout waiting for register bit.")
            return False

        time.sleep(poll_interval)


# Example usage: wait for bit 0 of register 150
# -------- Setup RealSense and YOLO ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_sensor = profile.get_device().first_depth_sensor()
real_depth_scale = depth_sensor.get_depth_scale()

fx, fy = color_intrinsics.fx, color_intrinsics.fy
cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
dist_coeffs = np.array(color_intrinsics.coeffs)  # k1,k2,p1,p2,k3
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])


#print("Depth scale (m/unit):", real_depth_scale)

model = YOLO(MODEL_WEIGHTS)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
def check_for_candy():
    frames = pipeline.wait_for_frames()
    color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    fx, fy = color_intrinsics.fx, color_intrinsics.fy
    cx, cy = color_intrinsics.ppx, color_intrinsics.ppy
    dist_coeffs = np.array(color_intrinsics.coeffs)  # k1,k2,p1,p2,k3
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

    results = model.predict(cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
                            imgsz=1280, conf=CONF_THRESHOLD, verbose=False)

    bbox = None
    results = model(color)
    # img_with_detection=results[0].plot()
    # cv2.imshow('Original', img_with_detection)
    bbox_rect = None  # fallback
    if results and results[0].obb is not None:
        obb = results[0].obb
        # Take first detection (or loop over all)
        xywhr = obb.xywhr[0].cpu().numpy()  # x_center, y_center, w, h, rotation (rad)
        x_c, y_c, w, h, rot = xywhr
        # Convert to rectangle (axis-aligned)
        x1 = int(x_c - w / 2)
        y1 = int(y_c - h / 2)
        x2 = int(x_c + w / 2)
        y2 = int(y_c + h / 2)
        bbox_rect = (x1, y1, x2, y2)

    if bbox_rect is None:
        # no detection
        print("No OBB detected this frame")
    else:
        x1, y1, x2, y2 = bbox_rect
        cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 255), 2)
    #
    x1, y1, x2, y2 = bbox_rect
    sx1, sy1, sx2, sy2 = shrink_bbox(x1, y1, x2, y2, SHRINK_FACTOR)
    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(color.shape[1] - 1, sx2), min(color.shape[0] - 1, sy2)

    # collect 3D points + pixel coords
    pts, pix = [], []
    for v in range(sy1, sy2, SAMPLE_STEP):
        for u in range(sx1, sx2, SAMPLE_STEP):
            d = depth[v, u] * real_depth_scale
            if d <= 0 or d > 3.0: continue
            p = deproject_pixel(u, v, d, depth_intrinsics)
            if p is not None:
                pts.append(p);
                pix.append((u, v))
    pts = np.array(pts);
    pix = np.array(pix)
    if pts.shape[0] < MIN_POINTS_FOR_PLANE:
        cv2.rectangle(color, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("frame", color)
        cv2.waitKey(1)

    normal, d, mask = ransac_plane(pts, MAX_RANSAC_ITERS, RANSAC_INLIER_THRESH)
    if mask is None:
        cv2.imshow("frame", color)
        cv2.waitKey(1)


    inliers, outliers = pix[mask], pix[~mask]
    for (u, v) in inliers: cv2.circle(color, (u, v), 1, (0, 255, 0), -1)
    for (u, v) in outliers: cv2.circle(color, (u, v), 1, (0, 0, 255), -1)

    pose_cam = compute_pose_from_points_center_only(pts[mask])
    pose_cam = pose_cam

    object_pose = return_robot_pose(pose_cam, camera_to_robot)
    # print('object_pose:', object_pose)
    # pose_robot = camera_to_robot @ pose_cam
    centroid = pose_cam[:3, 3]
    px = rs.rs2_project_point_to_pixel(color_intrinsics, [float(centroid[0]), float(centroid[1]), float(centroid[2])])
    cv2.circle(color, (int(px[0]), int(px[1])), 6, (255, 255, 0), -1)

    R = pose_cam[:3, :3]
    # Extract translation
    t = object_pose[:3, 3]
    # Store as [x, y, z, Rx, Ry, Rz]

    euler_deg = np.degrees(rotmat_to_euler_zyz(R))
    # print("Centroid (m):", centroid*1000)
    # print("Euler (deg):", euler_deg)
    # object_pose is a numpy array, e.g. [x, y, z, w]

    pose_list = [t[0], t[1], t[2], euler_deg[0], euler_deg[1], euler_deg[2]]
    return pose_list , color

# Thread-safe queue to send frames to the display
frame_queue = queue.Queue(maxsize=1)
# Start the display thread
t = threading.Thread(target=display_thread, daemon=True)
t.start()

try:
    while True:
        # intel realsense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Send frame to the display queue (overwrite if full)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(color_image)
        #print('start')
        ready = wait_for_register_bit(150, bit_index=0, poll_interval=0.05, timeout=5)

        if ready==1:
            try:
                pose_list,color = check_for_candy()
                print("Robot ready, sending pose...")
                # send your pose here
                if client.open():
                    # write multiple registers
                    # Convert float -> uint16

                    for i, val in enumerate(pose_list):
                        b = struct.pack('>f', val)  # float -> 4 bytes
                        hi, lo = struct.unpack('>HH', b)  # 2x16-bit registers
                        success = client.write_multiple_registers(128 + i * 2, [hi, lo])
                        if not success:
                            print(f"Failed to write registers {128 + i * 2} and {128 + i * 2 + 1}")
                    if success:
                        print("Pose Sent:", pose_list)
                        success = client.write_single_register(150, 2)
                    else:
                        print("Failed to write registers.")

            except:
                continue

            else:
                print("Unable to connect to Doosan Modbus TCP Slave.")
            cv2.imshow("frame", color)
        else:
            print("Robot not ready, aborting.")



        if cv2.waitKey(1) & 0xFF == 27: break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
