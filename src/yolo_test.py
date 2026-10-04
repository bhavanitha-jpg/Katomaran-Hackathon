import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/sample_video.mp4"
OUTPUT_PATH = "outputs/yolo_test.mp4"

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    print(f"Processed frame {frame_number}/{240}", end="\r")

cap.release()
out.release()

print("\nYOLO detection completed.")
print(f"Output saved to: {OUTPUT_PATH}")