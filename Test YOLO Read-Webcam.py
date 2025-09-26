import cv2
import numpy as np
import cv2.aruco as aruco
from robodk.robomath import Mat
from robodk.robolink import Robolink
from robodk.robolink import*
from ultralytics import YOLO
import cv2
import torch
import math
import pyrealsense2 as rs
# Load model
model = YOLO("C:/Users/skha/PycharmProjects/GolfPutty/lego_training/Red Lego63/weights/best.pt")
#RDK = Robolink()
#camera = RDK.Item('My Camera')
calibration_file = r"C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/RoboDK-Camera-Settings.yaml"
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
#camera_matrix = fs.getNode("Camera_Matrix").mat()
#robot=RDK.Item('Doosan Robotics H2515')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)
pipeline.start()
objects=[]
def get_robodk_camera_frame():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return color_image
def detect_objects_and_measure():
    """Detect ArUco marker, find contours, and display object dimensions."""

    frame = get_robodk_camera_frame()

    device = "cuda"
    model.to(device)  # Move the model to GPU if available
    results = model(frame, conf=0.75)
    img_with_boxes = results[0].plot()

    '''
    if frame is None:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids_aruco, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    # needed for accuracy improvements within 1 mm
    for corner in corners:
        cv2.cornerSubPix(
            gray,
            corner,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
        )
    if ids_aruco is None:
        print("No ArUco marker detected.")
        return

    aruco.drawDetectedMarkers(frame, corners, ids_aruco)
    px_to_mm = get_pixel_to_mm_ratio(corners)  # e.g., 0.25 mm/pixel

    print('px_to_mm:', px_to_mm)
    '''
    #obb.xyxyxyxy|  [n, 8]|  4 corner points (x1, y1, x2, y2, x3, y3, x4, y4)
    #obb.xywhr| [n, 5]| [x_center, y_center, width, height, angle_rad]
    #corners[4, 2] Pixel coordinates of all 4 corners

    # For each detection
    for result in results:
        obbs=result.obb
        #print(obbs)
        if result.obb:
            for obb in result.obb.xywhr.cpu().numpy():
                x_tl, y_tl, width, height, angle_rad = obb
                x_center = x_tl - width / 2
                y_center = y_tl - height / 2
                objects.append({
                    'bbox': ( width, height),
                    'center': (x_center, y_center),
                    'size': (16, 32),
                    'tvec:':np.array([x_center, y_center, width, height])
                })

    #rvec, tvec, image_points, object_points=tvec_rvec(objects)

                # Create robot target
    cv2.imshow('Original', img_with_boxes)
    cv2.waitKey(1)

    if key == ord('k'):

        files = glob.glob(os.path.join(save_folder, "capture_*.png"))
        n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
        filename = os.path.join(save_folder, f"capture_{n}.png")
        cv2.imwrite(filename, roi_resized)
        print(f"Saved ROI image: {filename}")
    #print(pose)
   # robot.MoveJ(pose)
    #results = model(frame, conf=0.7)
while(1):
    detect_objects_and_measure()