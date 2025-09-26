import cv2
import numpy as np
import cv2.aruco as aruco

from robodk.robomath import Mat
from robodk.robolink import *
board = cv2.aruco.CharucoBoard(
    size=(5,7),
    squareLength=0.035,
    markerLength=0.020,
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
)
picture_range=17
'''
RDK= Robolink()

Poses = [RDK.Item(f'Target {i}') for i in range(1, picture_range)]
for i in range(1, picture_range):
    globals()[f"Pose_{i}"] = RDK.Item(f"Target {i}")

camera= RDK.Item('Cali')
robot = RDK.Item("Doosan Robotics H2515")
robot.setSpeed(1000)  # mm/s
robot.setAcceleration(1000)  # mm/s²

if not camera.Valid():
    print('CAMERA NOT FOUND')
    exit()
'''
calibration_file = "C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/calibration.xml"

cap=cv2.VideoCapture(0)


def chaUco_Calibration(calibration_file):
    # Detection setup

    # Calibration data collection
    all_corners = []
    all_ids = []
    image_size = None
    calibration_complete = False
    detector = aruco.CharucoDetector(board)
    while True:
        #for i, pose in enumerate(Poses):
            #robot.MoveJ(pose)
            # Get frame
            ret, frames = cap.read()

            #bytes_img = RDK.Cam2D_Snapshot("", camera)
            #if not bytes_img:
            #    continue

            #nparr = np.frombuffer(bytes_img, np.uint8)
            #frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            #cv2.imwrite(f"C:/Users/skha/PycharmProjects/GolfPutty/CalibrationFiles/{i}.png", frame)
            if image_size is None:
                image_size = frames.shape[1], frames.shape[0]

            # Detection
            frame = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            # Detect ChArUco board
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(frame)

            if charuco_ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            #display_text = "Press 'c' to capture calibration frame"

            if charuco_ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                frame = cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

            #key = cv2.waitKey(1)
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print(f"Captured frame {len(all_corners)}")

            # Calibrate when enough frames (press 'f' to force)
            key = cv2.waitKey(1)
            if ((len(all_corners) >= picture_range)) and not calibration_complete:
                print("Calibrating...")
                ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                    all_corners, all_ids, board, image_size, None, None)

                if ret < 1:  # Good reprojection error

                    fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_WRITE)
                    if not fs.isOpened():
                        print("Error: Unable to open file for writing. Check file path and permissions.")
                        exit(1)  # Stop execution
                    fs.write("Camera_Matrix", mtx)
                    fs.write("Distortion_Coefficients", dist)
                    print("Calibration complete and uploaded to RoboDK!")
                    print(f"Reprojection error: {ret:.2f} pixels")
                    return
                else:
                    print(f"Poor calibration (error: {ret:.2f} px). Capture more frames.")

            # Display info
            #cv2.putText(frame, display_text, (10, 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #cv2.putText(frame, f"Frames captured: {len(all_corners)}/15", (10, 60),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('ChArUco Calibration', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    chaUco_Calibration( calibration_file)  # Only runs when executed directly