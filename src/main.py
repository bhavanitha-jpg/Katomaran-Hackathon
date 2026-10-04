import cv2
import json
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from database import (
    initialize_database,
    add_visitor,
    update_visitor,
    get_all_visitors
)

from logger import (
    initialize_logger,
    log_event
)


# ============================================================
# CONFIGURATION
# ============================================================

with open("config.json", "r") as f:
    config = json.load(f)

VIDEO_SOURCE = config["video_source"]
SKIP_FRAMES = config["detection_skip_frames"]
FACE_THRESHOLD = config["face_similarity_threshold"]
MIN_FACE_SIZE = config.get("min_face_size", 80)

OUTPUT_VIDEO = config["output_video"]

# Yellow entry/exit line
ROI_Y = 1080

# Number of frames before removing an inactive track
FRAME_EXIT_TIMEOUT = 30


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# FIND MATCHING VISITOR
# ============================================================

def find_matching_visitor(embedding):

    visitors = get_all_visitors()

    best_id = None
    best_similarity = -1.0

    for visitor in visitors:

        visitor_id = visitor[0]
        embedding_json = visitor[4]

        try:

            stored_embedding = np.array(
                json.loads(embedding_json),
                dtype=np.float32
            )

        except Exception:

            continue

        similarity = cosine_similarity(
            embedding,
            stored_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_id = visitor_id

    if best_similarity >= FACE_THRESHOLD:

        return best_id, best_similarity

    return None, best_similarity


# ============================================================
# SAVE FACE IMAGE
# ============================================================

def save_face_image(
    face_image,
    visitor_id,
    track_id,
    frame_number
):

    import os

    os.makedirs(
        "logs/entries",
        exist_ok=True
    )

    filename = (
        f"logs/entries/"
        f"visitor_{visitor_id}_"
        f"track_{track_id}_"
        f"frame_{frame_number}.jpg"
    )

    cv2.imwrite(
        filename,
        face_image
    )

    return filename


# ============================================================
# INITIALIZE DATABASE + LOGGER
# ============================================================

initialize_database()

initialize_logger()


# ============================================================
# LOAD YOLO
# ============================================================

print()
print("=" * 60)
print("Loading YOLO model...")
print("=" * 60)

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print()
print("=" * 60)
print("Loading InsightFace...")
print("=" * 60)

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("=" * 60)
print("Opening video...")
print("=" * 60)

cap = cv2.VideoCapture(
    VIDEO_SOURCE
)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    raise SystemExit


# ============================================================
# VIDEO INFORMATION
# ============================================================

FPS = cap.get(
    cv2.CAP_PROP_FPS
)

WIDTH = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

HEIGHT = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

TOTAL_FRAMES = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print()
print("Video information:")
print(f"FPS: {FPS}")
print(f"Resolution: {WIDTH} x {HEIGHT}")
print(f"Total frames: {TOTAL_FRAMES}")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():

    print(
        "ERROR: Could not create output video."
    )

    cap.release()

    raise SystemExit


# ============================================================
# TRACK INFORMATION
# ============================================================

# Track ID -> Visitor ID
track_to_visitor = {}

# Track ID -> Last frame seen
track_last_seen = {}

# Track ID -> Previous center Y position
track_previous_y = {}

# Track ID -> Last detected event
# Prevents repeated ENTRY/EXIT events
track_last_event = {}


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # ========================================================
    # DRAW ENTRY / EXIT LINE
    # ========================================================

    cv2.line(
        frame,
        (0, ROI_Y),
        (frame.shape[1], ROI_Y),
        (0, 255, 255),
        5
    )

    cv2.putText(
        frame,
        "ENTRY / EXIT LINE",
        (50, ROI_Y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 255),
        3
    )


    results = None


    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    if frame_number % SKIP_FRAMES == 0:

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            verbose=False
        )


        if (
            len(results) > 0
            and
            results[0].boxes is not None
            and
            results[0].boxes.id is not None
        ):

            result = results[0]

            boxes = result.boxes


            # =================================================
            # PROCESS EACH PERSON
            # =================================================

            for i in range(len(boxes)):

                # ---------------------------------------------
                # BOUNDING BOX
                # ---------------------------------------------

                xyxy = (
                    boxes
                    .xyxy[i]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )


                # ---------------------------------------------
                # TRACK ID
                # ---------------------------------------------

                track_id = int(
                    boxes
                    .id[i]
                    .item()
                )


                # ---------------------------------------------
                # PERSON CENTRE
                # ---------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ---------------------------------------------
                # UPDATE LAST SEEN
                # ---------------------------------------------

                track_last_seen[
                    track_id
                ] = frame_number


                # =================================================
                # FACE RECOGNITION
                # =================================================

                if track_id not in track_to_visitor:

                    crop_x1 = max(
                        0,
                        x1
                    )

                    crop_y1 = max(
                        0,
                        y1
                    )

                    crop_x2 = min(
                        WIDTH,
                        x2
                    )

                    crop_y2 = min(
                        HEIGHT,
                        y2
                    )


                    person_crop = frame[
                        crop_y1:crop_y2,
                        crop_x1:crop_x2
                    ]


                    if person_crop.size > 0:

                        faces = face_app.get(
                            person_crop
                        )


                        if len(faces) > 0:

                            # -------------------------------------
                            # LARGEST FACE
                            # -------------------------------------

                            face = max(
                                faces,
                                key=lambda f:
                                (
                                    f.bbox[2]
                                    -
                                    f.bbox[0]
                                )
                                *
                                (
                                    f.bbox[3]
                                    -
                                    f.bbox[1]
                                )
                            )


                            # -------------------------------------
                            # FACE SIZE
                            # -------------------------------------

                            face_width = (
                                face.bbox[2]
                                -
                                face.bbox[0]
                            )

                            face_height = (
                                face.bbox[3]
                                -
                                face.bbox[1]
                            )


                            # -------------------------------------
                            # MINIMUM FACE SIZE
                            # -------------------------------------

                            if (
                                face_width >= MIN_FACE_SIZE
                                and
                                face_height >= MIN_FACE_SIZE
                            ):

                                embedding = face.embedding


                                # ---------------------------------
                                # MATCH DATABASE
                                # ---------------------------------

                                matched_id, similarity = (
                                    find_matching_visitor(
                                        embedding
                                    )
                                )


                                # ---------------------------------
                                # EXISTING VISITOR
                                # ---------------------------------

                                if matched_id is not None:

                                    visitor_id = matched_id

                                    update_visitor(
                                        visitor_id
                                    )

                                    print(
                                        f"Track {track_id} "
                                        f"matched Visitor "
                                        f"{visitor_id} "
                                        f"(similarity: "
                                        f"{similarity:.3f})"
                                    )

                                    log_event(
                                        "RECOGNIZED",
                                        visitor_id,
                                        track_id,
                                        (
                                            f"similarity="
                                            f"{similarity:.3f}"
                                        )
                                    )


                                # ---------------------------------
                                # NEW VISITOR
                                # ---------------------------------

                                else:

                                    visitor_id = add_visitor(
                                        embedding
                                    )

                                    print(
                                        f"New Visitor "
                                        f"{visitor_id} "
                                        f"created for "
                                        f"Track {track_id} "
                                        f"(best similarity: "
                                        f"{similarity:.3f})"
                                    )

                                    log_event(
                                        "NEW_VISITOR",
                                        visitor_id,
                                        track_id,
                                        (
                                            f"best_similarity="
                                            f"{similarity:.3f}"
                                        )
                                    )


                                # ---------------------------------
                                # SAVE IDENTITY
                                # ---------------------------------

                                track_to_visitor[
                                    track_id
                                ] = visitor_id


                                # ---------------------------------
                                # SAVE FACE IMAGE
                                # ---------------------------------

                                save_face_image(
                                    person_crop,
                                    visitor_id,
                                    track_id,
                                    frame_number
                                )


                            else:

                                print(
                                    f"Track {track_id}: "
                                    f"face too small "
                                    f"({int(face_width)}x"
                                    f"{int(face_height)})"
                                )


                        else:

                            print(
                                f"Track {track_id}: "
                                f"No face detected"
                            )


                # =================================================
                # ACTUAL ENTRY / EXIT LINE CROSSING
                # =================================================

                if track_id in track_previous_y:

                    previous_y = (
                        track_previous_y[
                            track_id
                        ]
                    )


                    # ---------------------------------------------
                    # ABOVE -> BELOW
                    # ENTRY
                    # ---------------------------------------------

                    if (
                        previous_y < ROI_Y
                        and
                        center_y >= ROI_Y
                    ):

                        if (
                            track_last_event.get(track_id)
                            != "ENTRY"
                        ):

                            if track_id in track_to_visitor:

                                visitor_id = (
                                    track_to_visitor[
                                        track_id
                                    ]
                                )


                                log_event(
                                    "ENTRY",
                                    visitor_id,
                                    track_id,
                                    (
                                        "Crossed ROI "
                                        "downward"
                                    )
                                )


                                print(
                                    f"ENTRY -> "
                                    f"Visitor {visitor_id} "
                                    f"(Track {track_id})"
                                )


                                track_last_event[
                                    track_id
                                ] = "ENTRY"


                    # ---------------------------------------------
                    # BELOW -> ABOVE
                    # EXIT
                    # ---------------------------------------------

                    elif (
                        previous_y > ROI_Y
                        and
                        center_y <= ROI_Y
                    ):

                        if (
                            track_last_event.get(track_id)
                            != "EXIT"
                        ):

                            if track_id in track_to_visitor:

                                visitor_id = (
                                    track_to_visitor[
                                        track_id
                                    ]
                                )


                                log_event(
                                    "EXIT",
                                    visitor_id,
                                    track_id,
                                    (
                                        "Crossed ROI "
                                        "upward"
                                    )
                                )


                                print(
                                    f"EXIT -> "
                                    f"Visitor {visitor_id} "
                                    f"(Track {track_id})"
                                )


                                track_last_event[
                                    track_id
                                ] = "EXIT"


                # ---------------------------------------------
                # STORE CURRENT Y
                # ---------------------------------------------

                track_previous_y[
                    track_id
                ] = center_y


                # =================================================
                # DRAW PERSON
                # =================================================

                if track_id in track_to_visitor:

                    visitor_id = (
                        track_to_visitor[
                            track_id
                        ]
                    )

                    label = (
                        f"Visitor {visitor_id} | "
                        f"Track {track_id}"
                    )

                else:

                    label = (
                        f"Unknown | "
                        f"Track {track_id}"
                    )


                # ---------------------------------------------
                # PERSON BOX
                # ---------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    4
                )


                # ---------------------------------------------
                # LABEL
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(
                            40,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3
                )


                # ---------------------------------------------
                # CENTRE POINT
                # ---------------------------------------------

                cv2.circle(
                    frame,
                    (
                        center_x,
                        center_y
                    ),
                    8,
                    (255, 0, 0),
                    -1
                )


    # ========================================================
    # REMOVE LOST TRACKS
    # ========================================================

    disappeared_tracks = []


    for track_id in list(
        track_last_seen.keys()
    ):

        if (
            frame_number
            -
            track_last_seen[track_id]
            >
            FRAME_EXIT_TIMEOUT
        ):

            disappeared_tracks.append(
                track_id
            )


    for track_id in disappeared_tracks:

        # ---------------------------------------------
        # IMPORTANT:
        # We DO NOT call this an EXIT anymore.
        # EXIT is only caused by crossing the ROI line.
        # ---------------------------------------------

        if track_id in track_to_visitor:

            visitor_id = (
                track_to_visitor[
                    track_id
                ]
            )

            print(
                f"Track {track_id} "
                f"lost "
                f"(Visitor {visitor_id})"
            )

            # Remove tracking information
            del track_to_visitor[
                track_id
            ]


        if track_id in track_last_seen:

            del track_last_seen[
                track_id
            ]


        if track_id in track_previous_y:

            del track_previous_y[
                track_id
            ]


        if track_id in track_last_event:

            del track_last_event[
                track_id
            ]


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if frame_number % 30 == 0:

        print(
            f"Processed "
            f"{frame_number}/"
            f"{TOTAL_FRAMES} frames"
        )


# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()

writer.release()


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(
    f"Total frames processed: "
    f"{frame_number}"
)

print(
    f"Unique visitors: "
    f"{len(get_all_visitors())}"
)

print(
    f"Output video: "
    f"{OUTPUT_VIDEO}"
)

print("=" * 60)