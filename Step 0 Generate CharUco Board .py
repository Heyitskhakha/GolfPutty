import cv2
import cv2.aruco as aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# -----------------------------
# Parameters (real-world)
# -----------------------------
squares_x = 5     # number of squares along X
squares_y = 7     # number of squares along Y

square_length_mm = 35  # size of a chessboard square in millimeters
marker_length_mm = 25  # size of an ArUco marker inside a square in millimeters

# -----------------------------
# Create Charuco Board
# -----------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

board = aruco.CharucoBoard(
    size=(squares_x, squares_y),
    squareLength=square_length_mm,
    markerLength=marker_length_mm,
    dictionary=aruco_dict
)

# -----------------------------
# Generate Charuco Image (high resolution for PDF)
# -----------------------------
img_size = (squares_x * 200, squares_y * 200)  # 200 px per square (arbitrary, just for rendering)
img = board.generateImage(img_size)

# -----------------------------
# Save as PDF in real-world mm
# -----------------------------
pdf_filename = "charuco_board.pdf"

# Create a PDF canvas sized to fit the board in mm
c = canvas.Canvas(pdf_filename, pagesize=(
    squares_x * square_length_mm * mm,
    squares_y * square_length_mm * mm
))

# Convert the OpenCV image (numpy array) to something ReportLab can draw
# OpenCV gives BGR, so convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save temporary PNG (ReportLab needs a file)
tmp_file = "charuco_tmp.png"
cv2.imwrite(tmp_file, img_rgb)

# Draw image at 0,0 scaled to board size in mm
c.drawImage(tmp_file, 0, 0,
            width=squares_x * square_length_mm * mm,
            height=squares_y * square_length_mm * mm)

c.save()

print(f"Charuco board saved as {pdf_filename}")
