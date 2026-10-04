import os
import cv2
import json
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from database import (
    initialize_database,
    add_visitor,
    get_all_visitors,
    update_visitor,
    add_event,
    get_unique_visitor_count,
)

from logger import initialize_logger, log_event


# ============================================================
# CONFIGURATION
# ============================================================

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

VIDEO_SOURCE = config.get("video_source", "data/sample_video.mp4")
SKIP_FRAMES = config.get("detection_skip_frames", 5)
SIMILARITY_THRESHOLD = config.get("face_similarity_threshold", 0.60)
MIN_FACE_SIZE = config.get("min_face_size", 80)

DATABASE_PATH = config.get("database_path", "database/visitors.db")
ENTRY_LOG_DIR = config.get("entry_log_dir", "logs/entries")
EXIT_LOG_DIR = config.get("exit_log_dir", "logs/exits")
OUTPUT_VIDEO = config.get("output_video", "outputs/output.mp4")

EVENT_LOG = config.get("event_log", "logs/events.log")

# ROI line.
# For the current 3840x2160 video, 1080 is the middle horizontal line.
ROI_Y = 1080

# Remove tracking information after this many missing frames.
FRAME_EXIT_TIMEOUT = 30

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings."""

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def find_matching_visitor(embedding):
    """
    Compare the current face embedding with all registered visitors.

    Returns:
        visitor_id, similarity
    """

    visitors = get_all_visitors()

    best_id = None
    best_similarity = 0.0

    for visitor_id, first_seen, last_seen, visit_count, embedding_json in visitors:

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

    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_id, best_similarity

    return None, best_similarity


def get_largest_face(faces):
    """Return the largest detected face."""

    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            (face.bbox[2] - face.bbox[0]) *
            (face.bbox[3] - face.bbox[1])
        )
    )


def save_event_image(
    person_crop,
    face_crop,
    visitor_id,
    track_id,
    event_type
):
    """
    Save an image for an ENTRY or EXIT event.

    Preferred image:
        face crop

    Fallback:
        person crop
    """

    if event_type == "ENTRY":
        base_dir = ENTRY_LOG_DIR
    else:
        base_dir = EXIT_LOG_DIR

    date_folder = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, date_folder)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    filename = (
        f"visitor_{visitor_id}_"
        f"track_{track_id}_"
        f"{timestamp}.jpg"
    )

    image_path = os.path.join(output_dir, filename)

    image_to_save = face_crop

    if image_to_save is None or image_to_save.size == 0:
        image_to_save = person_crop

    if image_to_save is None or image_to_save.size == 0:
        return None

    success = cv2.imwrite(image_path, image_to_save)

    if not success:
        print(f"WARNING: Could not save event image: {image_path}")
        return None

    return image_path


def detect_current_face(app, person_crop):
    """
    Detect a face from the current person crop.

    Used when an ENTRY/EXIT event occurs so the event image
    represents the current crossing as closely as possible.
    """

    if person_crop is None or person_crop.size == 0:
        return None

    try:
        faces = app.get(person_crop)

        if not faces:
            return None

        face = get_largest_face(faces)

        if face is None:
            return None

        x1, y1, x2, y2 = map(int, face.bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(person_crop.shape[1], x2)
        y2 = min(person_crop.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face_width = x2 - x1
        face_height = y2 - y1

        if (
            face_width < MIN_FACE_SIZE
            or face_height < MIN_FACE_SIZE
        ):
            return None

        face_crop = person_crop[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        return face_crop

    except Exception as e:
        print(f"Face detection during event failed: {e}")
        return None


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 70)
print("KATOMARAN INTELLIGENT FACE TRACKER")
print("=" * 70)

initialize_database()
initialize_logger()

os.makedirs("outputs", exist_ok=True)
os.makedirs(ENTRY_LOG_DIR, exist_ok=True)
os.makedirs(EXIT_LOG_DIR, exist_ok=True)

# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")

# ============================================================
# LOAD INSIGHTFACE
# ============================================================

print("\nLoading InsightFace model...")

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
# OPEN VIDEO / RTSP STREAM
# ============================================================

print("\nOpening video source:")
print(VIDEO_SOURCE)

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video source: {VIDEO_SOURCE}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video FPS    : {fps:.2f}")
print(f"Video Width  : {width}")
print(f"Video Height : {height}")

# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_VIDEO}"
    )

# ============================================================
# TRACKING STATE
# ============================================================

track_to_visitor = {}

track_last_seen = {}

track_previous_y = {}

track_last_event = {}

track_face_crops = {}

track_person_crops = {}

# Persistent event state by visitor.
#
# This prevents:
# ENTRY -> ENTRY
# EXIT  -> EXIT
#
# for the same visitor without the opposite event occurring.
visitor_last_event = {}

frame_count = 0

# ============================================================
# START LOGGING
# ============================================================

log_event(
    "SYSTEM_START",
    visitor_id=0,
    track_id=None,
    details=(
        f"Video source={VIDEO_SOURCE}, "
        f"skip_frames={SKIP_FRAMES}, "
        f"threshold={SIMILARITY_THRESHOLD}, "
        f"min_face_size={MIN_FACE_SIZE}"
    )
)

# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------------------------------
    # DRAW ROI LINE
    # --------------------------------------------------------

    cv2.line(
        frame,
        (0, ROI_Y),
        (width, ROI_Y),
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

    # --------------------------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------------------------

    if frame_count % SKIP_FRAMES == 0:

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.25,
            verbose=False
        )

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):

            boxes = results[0].boxes

            ids = boxes.id.int().cpu().tolist()

            xyxy = boxes.xyxy.cpu().numpy()

            current_track_ids = set()

            for box, track_id in zip(xyxy, ids):

                track_id = int(track_id)

                current_track_ids.add(track_id)

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # ------------------------------------------------
                # PERSON CROP
                # ------------------------------------------------

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                track_person_crops[track_id] = person_crop.copy()

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                track_last_seen[track_id] = frame_count

                # ------------------------------------------------
                # NEW TRACK
                # ------------------------------------------------

                if track_id not in track_to_visitor:

                    log_event(
                        "TRACK_START",
                        visitor_id=0,
                        track_id=track_id,
                        details="New person track detected"
                    )

                    visitor_id = None

                    # --------------------------------------------
                    # FACE DETECTION
                    # --------------------------------------------

                    try:

                        faces = face_app.get(
                            person_crop
                        )

                    except Exception as e:

                        print(
                            f"Face detection failed "
                            f"for track {track_id}: {e}"
                        )

                        faces = []

                    face = get_largest_face(faces)

                    if face is not None:

                        face_width = (
                            face.bbox[2] -
                            face.bbox[0]
                        )

                        face_height = (
                            face.bbox[3] -
                            face.bbox[1]
                        )

                        # ----------------------------------------
                        # FACE SIZE CHECK
                        # ----------------------------------------

                        if (
                            face_width >= MIN_FACE_SIZE
                            and face_height >= MIN_FACE_SIZE
                        ):

                            embedding = face.embedding

                            # ------------------------------------
                            # SAVE FACE CROP FOR LATER EVENT
                            # ------------------------------------

                            fx1, fy1, fx2, fy2 = map(
                                int,
                                face.bbox
                            )

                            fx1 = max(0, fx1)
                            fy1 = max(0, fy1)

                            fx2 = min(
                                person_crop.shape[1],
                                fx2
                            )

                            fy2 = min(
                                person_crop.shape[0],
                                fy2
                            )

                            if (
                                fx2 > fx1
                                and fy2 > fy1
                            ):

                                face_crop = person_crop[
                                    fy1:fy2,
                                    fx1:fx2
                                ]

                                track_face_crops[
                                    track_id
                                ] = face_crop.copy()

                            # ------------------------------------
                            # EMBEDDING GENERATED
                            # ------------------------------------

                            log_event(
                                "EMBEDDING_GENERATED",
                                visitor_id=0,
                                track_id=track_id,
                                details=(
                                    f"Embedding generated "
                                    f"for face size "
                                    f"{int(face_width)}x"
                                    f"{int(face_height)}"
                                )
                            )

                            # ------------------------------------
                            # MATCH DATABASE
                            # ------------------------------------

                            matched_id, similarity = (
                                find_matching_visitor(
                                    embedding
                                )
                            )

                            if matched_id is not None:

                                visitor_id = matched_id

                                update_visitor(
                                    visitor_id
                                )

                                log_event(
                                    "RECOGNIZED",
                                    visitor_id=visitor_id,
                                    track_id=track_id,
                                    details=(
                                        f"similarity="
                                        f"{similarity:.3f}"
                                    )
                                )

                                print(
                                    f"Track {track_id} "
                                    f"recognized as Visitor "
                                    f"{visitor_id} "
                                    f"(similarity="
                                    f"{similarity:.3f})"
                                )

                            else:

                                visitor_id = add_visitor(
                                    embedding
                                )

                                log_event(
                                    "NEW_VISITOR",
                                    visitor_id=visitor_id,
                                    track_id=track_id,
                                    details=(
                                        f"New face registered; "
                                        f"best_similarity="
                                        f"{similarity:.3f}"
                                    )
                                )

                                print(
                                    f"Track {track_id}: "
                                    f"New Visitor "
                                    f"{visitor_id} "
                                    f"registered"
                                )

                        else:

                            print(
                                f"Track {track_id}: "
                                f"face too small for recognition "
                                f"({int(face_width)}x"
                                f"{int(face_height)})"
                            )

                    else:

                        print(
                            f"Track {track_id}: "
                            f"No face detected"
                        )

                    # --------------------------------------------
                    # STORE VISITOR ID
                    # --------------------------------------------

                    track_to_visitor[
                        track_id
                    ] = visitor_id

                    track_last_event[
                        track_id
                    ] = None

                # ------------------------------------------------
                # CURRENT VISITOR
                # ------------------------------------------------

                visitor_id = track_to_visitor.get(
                    track_id
                )

                previous_y = track_previous_y.get(
                    track_id
                )

                # ------------------------------------------------
                # ROI CROSSING
                # ------------------------------------------------

                if (
                    visitor_id is not None
                    and previous_y is not None
                ):

                    event_type = None

                    # Moving downward:
                    # ENTRY
                    if (
                        previous_y < ROI_Y
                        and center_y >= ROI_Y
                    ):
                        event_type = "ENTRY"

                    # Moving upward:
                    # EXIT
                    elif (
                        previous_y > ROI_Y
                        and center_y <= ROI_Y
                    ):
                        event_type = "EXIT"

                    if event_type is not None:

                        last_visitor_event = (
                            visitor_last_event.get(
                                visitor_id
                            )
                        )

                        # ----------------------------------------
                        # DUPLICATE EVENT PROTECTION
                        # ----------------------------------------

                        if (
                            last_visitor_event
                            == event_type
                        ):

                            print(
                                f"Ignored duplicate "
                                f"{event_type} for "
                                f"Visitor {visitor_id}"
                            )

                        else:

                            # ------------------------------------
                            # DETECT CURRENT FACE
                            # ------------------------------------

                            current_face_crop = (
                                detect_current_face(
                                    face_app,
                                    person_crop
                                )
                            )

                            if current_face_crop is None:

                                current_face_crop = (
                                    track_face_crops.get(
                                        track_id
                                    )
                                )

                            # ------------------------------------
                            # SAVE EVENT IMAGE
                            # ------------------------------------

                            image_path = (
                                save_event_image(
                                    person_crop=person_crop,
                                    face_crop=current_face_crop,
                                    visitor_id=visitor_id,
                                    track_id=track_id,
                                    event_type=event_type
                                )
                            )

                            # ------------------------------------
                            # DATABASE EVENT
                            # ------------------------------------

                            timestamp = (
                                datetime.now().isoformat()
                            )

                            add_event(
                                visitor_id=visitor_id,
                                track_id=track_id,
                                event_type=event_type,
                                timestamp=timestamp,
                                image_path=image_path,
                                details=(
                                    "Crossed ROI "
                                    f"{'downward' if event_type == 'ENTRY' else 'upward'}"
                                )
                            )

                            # ------------------------------------
                            # EVENT LOG
                            # ------------------------------------

                            log_event(
                                event_type,
                                visitor_id=visitor_id,
                                track_id=track_id,
                                details=(
                                    f"Crossed ROI "
                                    f"{'downward' if event_type == 'ENTRY' else 'upward'}; "
                                    f"image={image_path}"
                                )
                            )

                            print(
                                f"{event_type}: "
                                f"Visitor {visitor_id} "
                                f"| Track {track_id} "
                                f"| Image: {image_path}"
                            )

                            # ------------------------------------
                            # UPDATE EVENT STATE
                            # ------------------------------------

                            visitor_last_event[
                                visitor_id
                            ] = event_type

                            track_last_event[
                                track_id
                            ] = event_type

                # ------------------------------------------------
                # SAVE PREVIOUS POSITION
                # ------------------------------------------------

                track_previous_y[
                    track_id
                ] = center_y

                # ------------------------------------------------
                # DRAW TRACKING BOX
                # ------------------------------------------------

                if visitor_id is not None:

                    label = (
                        f"Visitor {visitor_id} "
                        f"| Track {track_id}"
                    )

                    box_color = (
                        0,
                        255,
                        0
                    )

                else:

                    label = (
                        f"Unknown "
                        f"| Track {track_id}"
                    )

                    box_color = (
                        0,
                        165,
                        255
                    )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    4
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    box_color,
                    3
                )

                # Center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    8,
                    (255, 0, 0),
                    -1
                )

            # ----------------------------------------------------
            # TRACK LOSS
            # ----------------------------------------------------

            for track_id in list(track_last_seen.keys()):

                if track_id in current_track_ids:
                    continue

                missing_frames = (
                    frame_count -
                    track_last_seen[track_id]
                )

                if missing_frames > FRAME_EXIT_TIMEOUT:

                    visitor_id = (
                        track_to_visitor.get(
                            track_id
                        )
                    )

                    log_event(
                        "TRACK_LOST",
                        visitor_id=(
                            visitor_id
                            if visitor_id is not None
                            else 0
                        ),
                        track_id=track_id,
                        details=(
                            f"Track lost after "
                            f"{missing_frames} frames"
                        )
                    )

                    print(
                        f"Track {track_id} lost"
                    )

                    track_last_seen.pop(
                        track_id,
                        None
                    )

                    track_to_visitor.pop(
                        track_id,
                        None
                    )

                    track_previous_y.pop(
                        track_id,
                        None
                    )

                    track_last_event.pop(
                        track_id,
                        None
                    )

                    track_face_crops.pop(
                        track_id,
                        None
                    )

                    track_person_crops.pop(
                        track_id,
                        None
                    )

    # ------------------------------------------------------------
    # DISPLAY FRAME INFORMATION
    # ------------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (50, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        f"Unique Visitors: {get_unique_visitor_count()}",
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )

    # ------------------------------------------------------------
    # WRITE OUTPUT
    # ------------------------------------------------------------

    writer.write(frame)

    # ------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------

    if frame_count % 30 == 0:

        print(
            f"Processed frame {frame_count}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

log_event(
    "SYSTEM_STOP",
    visitor_id=0,
    track_id=None,
    details=(
        f"Processing completed; "
        f"frames={frame_count}; "
        f"unique_visitors="
        f"{get_unique_visitor_count()}"
    )
)

print("\n" + "=" * 70)
print("PROCESSING COMPLETED")
print("=" * 70)

print(f"Total frames     : {frame_count}")
print(
    f"Unique visitors  : "
    f"{get_unique_visitor_count()}"
)
print(f"Output video     : {OUTPUT_VIDEO}")
print(f"Event log        : {EVENT_LOG}")
print(f"Entry images     : {ENTRY_LOG_DIR}")
print(f"Exit images      : {EXIT_LOG_DIR}")

print("=" * 70)