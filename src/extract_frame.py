import cv2

video_path = "data/sample_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Read the first frame
ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame")
    cap.release()
    exit()

output_path = "outputs/first_frame.jpg"

cv2.imwrite(output_path, frame)

print("First frame saved to:", output_path)

cap.release()