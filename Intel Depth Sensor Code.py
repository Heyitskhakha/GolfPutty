import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)

# Get intrinsics (once)
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
intr = depth_profile.get_intrinsics()
fx, fy, cx, cy = intr.fx, intr.fy, intr.ppy, intr.ppx

INCH_MM = 25.4

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Estimate pixel size at the average scene depth
        valid_depths = depth_image[depth_image > 0]
        if valid_depths.size == 0:
            continue
        avg_Z = np.mean(valid_depths)  # mm

        mm_per_pixel_x = avg_Z / fx
        mm_per_pixel_y = avg_Z / fy

        # Pixel size for ~1 inch
        px_per_inch_x = int(INCH_MM / mm_per_pixel_x)
        px_per_inch_y = int(INCH_MM / mm_per_pixel_y)

        h, w = depth_image.shape
        for y in range(0, h, px_per_inch_y):
            for x in range(0, w, px_per_inch_x):
                x2 = min(x + px_per_inch_x, w)
                y2 = min(y + px_per_inch_y, h)
                roi = depth_image[y:y2, x:x2]
                roi_valid = roi[roi > 0]
                if roi_valid.size > 0:
                    avg_depth = np.mean(roi_valid)
                    cv2.rectangle(depth_colormap, (x, y), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(depth_colormap,
                                f"{avg_depth/25.4:.1f}\"",  # convert mm → inches
                                (x + 5, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Depth Grid (inches)", depth_colormap)
        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
