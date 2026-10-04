import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/sample_video.mp4"
OUTPUT_PATH = "outputs/tracking_test.mp4"

MODEL_PATH = "yolo11n.pt"


print("Loading YOLO...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_number = 0

print("Starting ByteTrack...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    if frame_number % 30 == 0:
        print(
            f"Processed frame "
            f"{frame_number}"
        )

cap.release()
out.release()

print("Tracking completed.")
print("Output saved to:")
print(OUTPUT_PATH)