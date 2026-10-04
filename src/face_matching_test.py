import cv2
import numpy as np
from insightface.app import FaceAnalysis

from database import (
    initialize_database,
    add_visitor,
    get_all_visitors,
    update_visitor
)


VIDEO_PATH = "data/sample_video.mp4"

# Similarity threshold
SIMILARITY_THRESHOLD = 0.45


def cosine_similarity(embedding1, embedding2):

    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(
        np.dot(embedding1, embedding2)
        / (norm1 * norm2)
    )


print("Initializing database...")
initialize_database()

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

ret, frame = cap.read()

if not ret:
    print("ERROR: Could not read frame")
    cap.release()
    exit()

print("Detecting faces...")

faces = app.get(frame)

print("Faces detected:", len(faces))

existing_visitors = get_all_visitors()

print("Existing visitors:", len(existing_visitors))


for i, face in enumerate(faces):

    embedding = face.embedding

    print(f"\nFace {i + 1}")

    if embedding is None:
        print("No embedding found.")
        continue

    best_match_id = None
    best_similarity = 0.0

    for visitor in existing_visitors:

        visitor_id = visitor[0]
        stored_embedding = np.array(
            __import__("json").loads(visitor[4])
        )

        similarity = cosine_similarity(
            embedding,
            stored_embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_id = visitor_id

    if (
        best_match_id is not None
        and best_similarity >= SIMILARITY_THRESHOLD
    ):

        print(
            f"Matched existing visitor "
            f"{best_match_id}"
        )

        print(
            f"Similarity: "
            f"{best_similarity:.4f}"
        )

        update_visitor(best_match_id)

    else:

        visitor_id = add_visitor(embedding)

        print(
            f"New visitor created: "
            f"{visitor_id}"
        )

        if best_match_id is not None:
            print(
                f"Best similarity was "
                f"{best_similarity:.4f}"
            )

        existing_visitors = get_all_visitors()


cap.release()

print("\nFace matching test completed.")