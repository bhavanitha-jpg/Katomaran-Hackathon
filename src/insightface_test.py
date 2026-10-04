import cv2
from insightface.app import FaceAnalysis

VIDEO_PATH = "data/sample_video.mp4"
OUTPUT_PATH = "outputs/insightface_test.jpg"

# Load InsightFace
print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully.")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Read first frame
ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame")
    cap.release()
    exit()

print("Running face detection...")

# Detect faces
faces = app.get(frame)

print("Faces detected:", len(faces))

# Draw face boxes
for i, face in enumerate(faces):

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        f"Face {i + 1}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Check embedding
    if face.embedding is not None:
        print(
            f"Face {i + 1}: "
            f"embedding shape = {face.embedding.shape}"
        )

# Save result
cv2.imwrite(OUTPUT_PATH, frame)

print("Annotated image saved to:")
print(OUTPUT_PATH)

cap.release()