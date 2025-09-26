import cv2
import numpy as np
import math
import os
import glob

# Folder to save images
save_folder = "saved_legos"
os.makedirs(save_folder, exist_ok=True)

target_size = 640  # ROI size
step = 20  # ROI movement step

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Set webcam resolution to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Initial ROI top-left corner
roi_x, roi_y = 640, 220  # roughly centered

def pad_to_square_keep(image, target_size=640):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Draw movable ROI rectangle on full frame
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + target_size, roi_y + target_size), (0, 255, 0), 2)

    # Extract ROI for processing
    roi = frame[roi_y:roi_y + target_size, roi_x:roi_x + target_size].copy()

    # Convert ROI to gray for contour detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours in ROI
    contours, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours inside ROI for preview
    roi_display = roi.copy()
    if contours:
        cv2.drawContours(roi_display, contours, -1, (0, 0, 255), 2)

    cv2.imshow("ROI Preview", roi_display)
    cv2.imshow("Full 1080p View", frame)

    key = cv2.waitKey(1) & 0xFF

    # Move ROI
    if key == ord('w'):  # up
        roi_y = max(roi_y - step, 0)
    elif key == ord('s'):  # down
        roi_y = min(roi_y + step, frame.shape[0] - target_size)
    elif key == ord('a'):  # left
        roi_x = max(roi_x - step, 0)
    elif key == ord('d'):  # right
        roi_x = min(roi_x + step, frame.shape[1] - target_size)

    # Save ROI
    elif key == ord('k'):
        roi_resized = pad_to_square_keep(roi, target_size)
        files = glob.glob(os.path.join(save_folder, "capture_*.png"))
        n = max([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in files], default=0) + 1
        filename = os.path.join(save_folder, f"capture_{n}.png")
        cv2.imwrite(filename, roi_resized)
        print(f"Saved ROI image: {filename}")

    # Exit
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
