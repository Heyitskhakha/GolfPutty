import cv2
import numpy as np
import cv2.aruco as aruco

from robodk.robomath import Mat
from robodk.robolink import *
import math
from robodk.robolink import *
from robodk.robodialogs import *
import cv2 as cv
import numpy as np
import glob
import pyrealsense2 as rs
picture_range=15
#RDK= Robolink()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
#Poses = [RDK.Item(f'Target {i}') for i in range(1, picture_range)]
pipeline.start()
#camera_item= RDK.Item('My Camera')
#robot = RDK.Item("Doosan Robotics H2515")
#robot.setSpeed(1000)  # mm/s
#robot.setAcceleration(1000)  # mm/s²
#Lego=RDK.Item('Lego')
#if not camera_item.Valid():
#    print('CAMERA NOT FOUND')
#    exit()
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
    #if not roi_selected:
    #    print("No ROI selected.")
    #    return

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
    """
    Resize an image (ROI) to a square target size while keeping aspect ratio.
    The image is extended or cropped proportionally instead of padding black borders.
    """
    h, w = image.shape[:2]

    # Compute scale to fit target size
    scale = max(target_size / w, target_size / h)

    # Compute new size in original image coordinates
    new_w = math.ceil(target_size / scale)
    new_h = math.ceil(target_size / scale)

    # Center coordinates
    cx, cy = w // 2, h // 2

    # Compute ROI coordinates
    x1 = max(cx - new_w // 2, 0)
    y1 = max(cy - new_h // 2, 0)
    x2 = min(cx + new_w // 2, w)
    y2 = min(cy + new_h // 2, h)

    # Crop and resize
    roi = image[y1:y2, x1:x2]
    resized_roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return resized_roi

    # Save the image



calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"
output_dir="C:/Users/skha/PycharmProjects/GolfPutty/Pictures for Teaching/"
save_folder = output_dir
#cv2.namedWindow("Live Stream")
#cv2.setMouseCallback("Live Stream", select_roi)
start_deg=0
while(1):
    # Retrieve the image by socket

    #depth_frame = frames.get_depth_frame()

    # Convert to numpy
    #depth_image = np.asanyarray(depth_frame.get_data())

    #intel realsense
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

    #RoboDK CAMERA
    #bytes_img = camera_item.RDK().Cam2D_Snapshot('', camera_item)
    #nparr = np.frombuffer(bytes_img, np.uint8)
    #Img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    #display_frame = Img.copy()
    #gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
    # Draw rectangle if selecting

    # Find contours
    contours, ids = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500  # ignore tiny contours (noise)
    max_area = 5000  # ignore very large contours

    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered_contours.append(cnt)

    # Inside your main loop, after getting color_image and gray
    h_img, w_img = gray.shape
    center_x, center_y = w_img // 2, h_img // 2

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    min_area = 1
    max_area = 500000
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered_contours.append(cnt)

    # Find the contour closest to the center
    best_contour = None
    min_distance = float('inf')
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            best_contour = cnt

    # If we found a Lego contour near center
    if best_contour is not None:
        # Draw it on the image
        cv2.drawContours(color_image, [best_contour], -1, (0, 255, 0), 20)

        # Crop bounding box around contour
        x, y, w, h = cv2.boundingRect(best_contour)
        roi = color_image[y:y + h, x:x + w]

        # Resize/crop proportionally to square (using your pad_to_square function)
        roi_resized = pad_to_square(roi, target_size=480)

        # Optional: show the cropped Lego ROI
        cv2.imshow("Lego ROI", roi_resized)

    # Show the full frame with contours
    cv2.imshow('Contours', color_image)
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('k') or key == ord('K'):  # Press 'K' to capture
        save_roi()
    print('Ready for Picture')
    '''
    cv2.rectangle(display_frame, (10, 10), (110, 40), (0, 0, 255), -1)  # Clear
    cv2.putText(display_frame, "Clear ROI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(display_frame, (120, 10), (220, 40), (0, 255, 0), -1)  # Save
    cv2.putText(display_frame, "Save ROI", (125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    if cropping or roi_selected:
        cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    '''

    color_image =gray
    frame_size = gray.shape[::-1]
    # If found, add object points, image points (after refining them)
    # Find the chessboard corners
    # Show live stream
    #cv2.imshow("Live Stream", display_frame)
    #cv.imshow('Original image', Img)


    #key = cv.waitKey(1) & 0xFF



    #cv.waitKey(1)