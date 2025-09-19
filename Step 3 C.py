import cv2 as cv
import numpy as np

# Match exactly what you used during capture
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
board = cv.aruco.CharucoBoard((5, 7), 0.035, 0.021, aruco_dict)
detector = cv.aruco.CharucoDetector(board)

img = cv.imread("C:/Users/skha/PycharmProjects/GolfPutty/Hand-Eye-Data/0.png")
if img is None:
    print("Image not loaded")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

if charuco_ids is not None:
    print(f"✅ Detected {len(charuco_ids)} ChArUco corners")
    cv.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
    cv.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
else:
    print("❌ No ChArUco detected")

cv.imshow("Detection", img)
cv.waitKey(0)
