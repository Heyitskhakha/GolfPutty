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
import pyrealsense2 as rs
picture_range=15
RDK= Robolink()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
Poses = [RDK.Item(f'Target {i}') for i in range(1, picture_range)]

camera_item= RDK.Item('My Camera')
robot = RDK.Item("Doosan Robotics H2515")
robot.setSpeed(1000)  # mm/s
robot.setAcceleration(1000)  # mm/s²

if not camera_item.Valid():
    print('CAMERA NOT FOUND')
    exit()
# Global variables
roi_selected = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False

# Mouse callback
def select_roi(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, roi_selected

    # Check buttons first
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 40:  # Clear button
            clear_roi()
            return
        elif 120 <= x <= 220 and 10 <= y <= 40:  # Save button
            save_roi()
            return

        # If not clicking a button, start ROI
        x_start, y_start = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP and cropping:
        x_end, y_end = x, y
        cropping = False
        roi_selected = True
# Clear the ROI
def clear_roi():
    global roi_selected, x_start, y_start, x_end, y_end
    roi_selected = False
    x_start = y_start = x_end = y_end = 0
    print("ROI cleared.")
# Save the ROI
def save_roi():
    global save_counter
    if not roi_selected:
        print("No ROI selected.")
        return

    x1, y1 = min(x_start, x_end), min(y_start, y_end)
    x2, y2 = max(x_start, x_end), max(y_start, y_end)
    roi = Img[y1:y2, x1:x2]
    roi_resized = pad_to_square(roi)
    files = glob.glob(os.path.join(save_folder, "capture_*.png"))
    n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
    filename = os.path.join(save_folder, f"capture_{n}.png")
    # Save the image
    cv.imwrite(filename, roi_resized)
    print(f"Saved image: {filename}")

def pad_to_square(image, target_size=480):
    h, w = image.shape[:2]

    # Calculate padding
    delta_w = target_size - w
    delta_h = target_size - h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Apply padding (black background)
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

    # Save the image



calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"
output_dir="C:/Users/skha/PycharmProjects/GolfPutty/Pictures for Teaching/"
save_folder = output_dir
cv2.namedWindow("Live Stream")
cv2.setMouseCallback("Live Stream", select_roi)
while(1):
    # Retrieve the image by socket

    #depth_frame = frames.get_depth_frame()

    # Convert to numpy
    #depth_image = np.asanyarray(depth_frame.get_data())

    #intel realsense
    #frames = pipeline.wait_for_frames()
    #color_frame = frames.get_color_frame()
    #color_image = np.asanyarray(color_frame.get_data())
    #gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

    #RoboDK CAMERA
    bytes_img = camera_item.RDK().Cam2D_Snapshot('', camera_item)
    nparr = np.frombuffer(bytes_img, np.uint8)
    Img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    display_frame = Img.copy()
    gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    # Draw rectangle if selecting

    cv2.rectangle(display_frame, (10, 10), (110, 40), (0, 0, 255), -1)  # Clear
    cv2.putText(display_frame, "Clear ROI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(display_frame, (120, 10), (220, 40), (0, 255, 0), -1)  # Save
    cv2.putText(display_frame, "Save ROI", (125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if cropping or roi_selected:
        cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    color_image =gray
    frame_size = gray.shape[::-1]
    # If found, add object points, image points (after refining them)
    # Find the chessboard corners
    # Show live stream
    cv2.imshow("Live Stream", display_frame)
    #cv.imshow('Original image', Img)


    key = cv.waitKey(1) & 0xFF



    cv.waitKey(1)