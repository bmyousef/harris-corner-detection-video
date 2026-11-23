"""
Lab 4 - Harris Corner Detection on video
Usage: python harris_corner_detection_video.py
Place harris_corner_detection_image_V.mp4 in the same folder as this script.
Set USE_WEBCAM = True to use the webcam instead.
Press 'q' to quit.
"""
import os
import cv2
import numpy as np

# ---------- Parameters ----------
USE_WEBCAM = False
BLOCK_SIZE = 2
APERTURE_SIZE = 3
K = 0.04
THRESH_RATIO = 0.05
RADIUS = 3
SCALE = 0.5           # resize frames for speed / to fit the screen

here = os.path.dirname(os.path.abspath(__file__))
source = 0 if USE_WEBCAM else os.path.join(here, "harris_corner_detection_image_V.mp4")

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video source: {source}")

saved = False
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    frame_no += 1

    frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    original = frame.copy()

    gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    dst = cv2.cornerHarris(gray, BLOCK_SIZE, APERTURE_SIZE, K)
    dst = cv2.dilate(dst, None)

    ys, xs = np.where(dst > THRESH_RATIO * dst.max())
    for x, y in zip(xs, ys):
        cv2.circle(frame, (int(x), int(y)), RADIUS, (0, 0, 255), 1)

    # Save one before/after screenshot pair (frame 60) for submission
    if frame_no == 60 and not saved:
        cv2.imwrite(os.path.join(here, "video_before.png"), original)
        cv2.imwrite(os.path.join(here, "video_after.png"), frame)
        saved = True

    cv2.imshow("Original Video", original)
    cv2.imshow("Harris Corners - Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
