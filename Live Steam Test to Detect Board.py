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

# Start streaming
#pipeline.start(config)
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
output_dir="C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/"
save_folder = output_dir
SQUARES_X = 10  # number of squares along the X axis
SQUARES_Y = 6  # number of squares along the Y axis
PATTERN = (SQUARES_X-1, SQUARES_Y-1)
SQUARE_LENGTH = 25  # mm, length of one square
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

    gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    color_image =gray
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
        cv.drawChessboardCorners(color_image, PATTERN, rcorners, ret)
        cv.putText(color_image, f"Chessboard detected! ({len(img_points)} frames)",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv.putText(color_image, "No chessboard detected",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow('Original image', color_image)
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('k') or key == ord('K'):  # Press 'K' to capture
        # Generate a timestamped filename
        files = glob.glob(os.path.join(save_folder, "capture_*.png"))
        n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
        filename = os.path.join(save_folder, f"capture_{n}.png")

        # Save the image
        cv.imwrite(filename, gray)
        print(f"Saved image: {filename}")


    cv.waitKey(1)