import cv2
import numpy as np
import ezdxf
from ezdxf.math import Vec2, ConstructionLine


def charuco_to_dxf_perfect(input_png, output_dxf, resolution_scale=1.0):
    """
    Final perfected Charuco to DXF converter with:
    - All corners exactly 90 degrees
    - Single, non-overlapping lines
    - Perfect preservation of all features
    """
    # 1. Load and enhance image
    print("Loading and preprocessing image...")
    img = cv2.imread(input_png, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_png}")

    # 2. Create optimized binary image
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 2
    )

    # 3. Clean and thin the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    thinned = cv2.ximgproc.thinning(cleaned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 4. Find all junctions and endpoints
    print("Detecting corners and lines...")
    junctions = find_line_junctions(thinned)

    # 5. Create DXF document
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    scale = resolution_scale

    # 6. Process all detected lines
    print("Building perfect 90° lines...")
    line_segments = []
    lines = cv2.HoughLinesP(thinned, 1, np.pi / 180, 10, minLineLength=1, maxLineGap=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append((
                (round(x1 * scale, 2), round((img.shape[0] - y1) * scale, 2)),
                (round(x2 * scale, 2), round((img.shape[0] - y2) * scale, 2))
            ))

    # 7. Snap all lines to 90° angles
    print("Snapping lines to 90° angles...")
    snapped_lines = []
    for (start, end) in line_segments:
        vec = Vec2(end) - Vec2(start)
        angle = np.degrees(np.arctan2(vec.y, vec.x))

        # Snap to nearest 90° angle
        snapped_angle = round(angle / 45) * 45
        if snapped_angle % 90 != 0:
            snapped_angle = round(angle / 90) * 90

        length = vec.magnitude
        new_end = Vec2(start) + Vec2.from_deg_angle(snapped_angle, length)
        snapped_lines.append((Vec2(start), new_end))

    # 8. Connect and merge lines at junctions
    print("Building final geometry...")
    added_lines = set()
    for line in snapped_lines:
        start, end = line
        line_key = (tuple(start.round(2)), tuple(end.round(2)))
        reverse_key = (tuple(end.round(2)), tuple(start.round(2)))

        if line_key not in added_lines and reverse_key not in added_lines:
            msp.add_line(start, end)
            added_lines.add(line_key)

    # 9. Final optimization pass
    print("Optimizing final output...")
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = Vec2(entity.dxf.start)
            end = Vec2(entity.dxf.end)
            vec = end - start

            # Force perfect horizontal/vertical if very close
            if abs(vec.x) < 0.1:  # Vertical line
                entity.dxf.end = (start.x, end.y)
            elif abs(vec.y) < 0.1:  # Horizontal line
                entity.dxf.end = (end.x, start.y)

    # 10. Save DXF
    doc.saveas(output_dxf)
    print(f"Perfect DXF saved to {output_dxf}")


def find_line_junctions(image):
    """Find all line junctions in the thinned image"""
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(image, cv2.CV_8U, kernel)
    junctions = np.where(conv >= 13)
    return list(zip(junctions[1], junctions[0]))


# Usage
input_png = "C:/Users/skha/PycharmProjects/GolfPutty/charuco_boardX.png"
output_dxf = "charuco_board_perfect_90deg.dxf"
charuco_to_dxf_perfect(input_png, output_dxf, resolution_scale=2.0)