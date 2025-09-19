import cv2
import numpy as np
import cv2.aruco as aruco

from robodk.robomath import Mat
from robodk.robolink import *

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)


board = cv2.aruco.CharucoBoard(
    size=(5,7),
    squareLength=0.035,
    markerLength=0.021,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
)
# Prepare calibration
all_corners = []
all_ids = []
image_size = None


camera_offset = np.array([0,0,50])
RDK= Robolink()
camera= RDK.Item('My Camera')
robot = RDK.Item("Doosan Robotics H2515")

def chaUco_Calibration():
    # Detection setup

    detector = aruco.CharucoDetector(board)
    # Calibration data collection
    all_corners = []
    all_ids = []
    image_size = None
    calibration_complete = False

    while True:

            bytes_img = RDK.Cam2D_Snapshot("", camera)
            if not bytes_img:
                continue

            nparr = np.frombuffer(bytes_img, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('ChArUco test', frame)
            if image_size is None:
                image_size = frame.shape[1], frame.shape[0]

            # Detect ChArUco board
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

            if charuco_ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                #display_text = f"Markers: {len(ids)} | Press 'c' to capture"

                # Store calibration data when 'c' pressed
                #key = cv2.waitKey(1)
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print(f"Captured frame {len(all_corners)}")

            else:
                print('NONE')

            #cv2.putText(frame, display_text, (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ChArUco Calibration', frame)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    chaUco_Calibration()  # Only runs when executed directly
