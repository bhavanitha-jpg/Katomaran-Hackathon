import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/sample_video.mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read first frame")
    cap.release()
    exit()

print("Running YOLO on first frame...")

results = model(frame, conf=0.10, verbose=False)

result = results[0]

if result.boxes is None or len(result.boxes) == 0:
    print("NO DETECTIONS FOUND")
else:
    print("Detections found:", len(result.boxes))

    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Detection {i + 1}: "
            f"class={class_id}, "
            f"name={model.names[class_id]}, "
            f"confidence={confidence:.2f}"
        )

    annotated = result.plot()

    cv2.imwrite(
        "outputs/yolo_diagnostic.jpg",
        annotated
    )

    print("Annotated image saved to:")
    print("outputs/yolo_diagnostic.jpg")

cap.release()