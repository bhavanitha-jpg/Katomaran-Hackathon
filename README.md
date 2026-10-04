# Katomaran Intelligent Face Tracker

## Hackathon Project

An AI-based intelligent visitor tracking and face recognition system that detects, tracks, recognizes, auto-registers, and counts unique visitors from a video stream.

The system combines **YOLO11n** for person detection, **ByteTrack** for multi-object tracking, **InsightFace** for face recognition and embeddings, **SQLite** for persistent storage, **ROI-based entry/exit detection**, timestamped event images, and structured event logging.

---

# 1. Problem Statement

The objective of this project is to build an intelligent face tracking and visitor counting system that can:

- Detect people from a video stream.
- Track people across frames.
- Detect and recognize faces.
- Automatically register previously unseen faces.
- Assign a unique visitor ID.
- Re-identify previously registered visitors.
- Track visitor movement.
- Detect visitor ENTRY and EXIT using an ROI line.
- Store visitor information in a local database.
- Log recognition, registration, tracking, entry, and exit events.
- Save timestamped evidence images for ENTRY and EXIT events.
- Maintain a unique visitor count.

The system is developed and tested using the provided sample video and is designed so that the input can also be replaced with an RTSP camera stream through the configuration file.

---

# 2. Key Features

- YOLO11n-based person detection.
- ByteTrack-based multi-object tracking.
- InsightFace Buffalo_L face detection and recognition.
- Face embedding generation.
- Cosine similarity-based face matching.
- Automatic registration of new visitors.
- Persistent visitor IDs using SQLite.
- Unique visitor counting.
- ROI-based ENTRY/EXIT detection.
- Timestamped ENTRY evidence images.
- Timestamped EXIT evidence images.
- Date-based event image folders.
- SQLite event records.
- Structured event logging.
- Tracking start and tracking lost logging.
- Embedding generation logging.
- Recognition logging.
- Configurable frame skipping.
- Configurable face similarity threshold.
- Configurable minimum face size.
- Local video support.
- RTSP stream support through configuration.
- Duplicate same-direction event protection.

---

# 3. System Architecture


                    Video / RTSP Stream
                            |
                            v
                  OpenCV VideoCapture
                            |
                            v
                  YOLO11n Detection
                            |
                            v
                   Person Detection
                            |
                            v
                    ByteTrack Tracking
                            |
                            v
                       Track ID
                            |
                            v
                 Person Crop / Face Crop
                            |
                            v
                  InsightFace / ArcFace
                            |
                            v
                    Face Embedding
                            |
                            v
                 +----------------------+
                 |    SQLite Database   |
                 |                      |
                 | Existing Visitor     |
                 |       -> Match       |
                 |                      |
                 | New Visitor          |
                 |       -> Register    |
                 +----------------------+
                            |
                            v
                       ROI Line
                      /       \
                     /         \
                  ENTRY        EXIT
                    |            |
                    v            v
             Evidence Image  Evidence Image
                    |            |
                    +------v-----+
                           |
                           v
                  SQLite Events Table
                           |
                           v
                     events.log
4. Overall Processing Pipeline
Video / RTSP
     |
     v
YOLO11n
     |
     v
Person Detection
     |
     v
ByteTrack
     |
     v
Track ID
     |
     v
Person Crop
     |
     v
InsightFace Face Detection
     |
     v
Face Embedding
     |
     +----------------------+
     |                      |
     v                      v
Existing Face          New Face
     |                      |
     v                      v
Recognize Visitor      Register Visitor
     |                      |
     +----------+-----------+
                |
                v
          ROI Crossing
          /          \
       ENTRY          EXIT
         |              |
         v              v
     Save Image      Save Image
         |              |
         +-------+------+
                 |
                 v
           SQLite Events
                 |
                 v
            Event Logs
5. Technologies Used
Programming Language
Python
Computer Vision
OpenCV
Object Detection
YOLO11n
Object Tracking
ByteTrack
Face Recognition
InsightFace
Buffalo_L model
Face embeddings
Cosine similarity matching
Database
SQLite
Configuration
JSON
Logging
Structured local event log
Timestamped event images
SQLite event records
6. Project Structure
Katomaran_Hackathon/
│
├── data/
│   └── sample_video.mp4
│
├── database/
│   └── visitors.db
│
├── logs/
│   ├── entries/
│   │   └── YYYY-MM-DD/
│   │
│   ├── exits/
│   │   └── YYYY-MM-DD/
│   │
│   └── events.log
│
├── models/
│
├── outputs/
│   └── output.mp4
│
├── src/
│   ├── main.py
│   ├── database.py
│   ├── logger.py
│   ├── entry_exit.py
│   ├── extract_frame.py
│   ├── face_matching_test.py
│   ├── insightface_test.py
│   ├── test_video.py
│   ├── tracking_test.py
│   ├── yolo_diagnostic.py
│   └── yolo_test.py
│
├── config.json
├── requirements.txt
├── yolo11n.pt
└── README.md

Runtime-generated files such as the SQLite database, logs, processed output video, and event images are excluded from version control where appropriate.

7. YOLO11n Person Detection

YOLO11n is used as the primary object detection model.

The system processes the person class from the YOLO detections.

For every detected person, the system obtains a bounding box:

x1, y1, x2, y2

The detected person crop is then used for the face recognition stage.

YOLO11n was selected because its lightweight architecture is suitable for video-based computer vision applications and provides a practical balance between detection capability and computational cost.

8. ByteTrack Tracking

ByteTrack is used after person detection to maintain a tracking identity across frames.

Example tracking IDs:

Track 21
Track 33
Track 54

A tracking ID allows the system to follow a person across multiple frames instead of treating every detection as a completely new object.

The tracker also provides movement information that is used by the ROI-based ENTRY/EXIT logic.

The implementation uses ByteTrack through the YOLO tracking interface.

9. Face Detection and Recognition

InsightFace is used for face detection and face embedding generation.

For a newly detected person track:

The person bounding box is cropped.
InsightFace searches for faces within the crop.
The largest suitable face is selected.
A face embedding is generated.
The embedding is compared against stored visitor embeddings.

The project uses InsightFace rather than the face_recognition library, in accordance with the hackathon requirements.

10. Face Embedding and Matching

The system represents a detected face using a numerical embedding.

The embedding is compared with stored visitor embeddings using cosine similarity.

The recognition flow is:

Current Face
     |
     v
Face Embedding
     |
     v
Compare with Stored Embeddings
     |
     +--------------------------+
     |                          |
Similarity >= Threshold   Similarity < Threshold
     |                          |
     v                          v
Existing Visitor           New Visitor
     |                          |
     v                          v
Recognize ID                Register ID

The current configuration uses:

"face_similarity_threshold": 0.60

A minimum face-size filter is also used:

"min_face_size": 80

This prevents very small face detections from being directly used for recognition.

Recognition quality can vary depending on face size, pose, lighting, occlusion, camera angle, and image quality.

11. Automatic Visitor Registration

When a detected face does not sufficiently match an existing visitor, the system automatically creates a new visitor record.

Example:

Visitor 1
Visitor 2
Visitor 3
...
Visitor 31

The visitor database stores the face embedding together with visitor metadata.

When a later face sufficiently matches an existing embedding, the system re-identifies the existing visitor rather than intentionally creating a new visitor record.

This provides automatic visitor registration without requiring manual visitor input.

12. Unique Visitor Counting

The unique visitor count is obtained from the visitor records stored in SQLite.

For the latest end-to-end test:

Total frames processed : 240
Unique visitors        : 31

The unique visitor count can be retrieved directly from the database.

Re-identification uses the existing visitor ID when the face similarity passes the configured threshold.

13. ROI-Based Entry and Exit Detection

A horizontal ROI line is used as the logical entrance/exit boundary.

For the current sample video:

ROI_Y = 1080

The sample video resolution is:

3840 x 2160

The ROI line is displayed in the processed output video as:

ENTRY / EXIT LINE

The movement logic is:

Previous Position Above ROI
          |
          | Downward Crossing
          v
        ENTRY

and:

Previous Position Below ROI
          |
          | Upward Crossing
          v
         EXIT

This allows the system to determine directional movement through the configured region.

The ROI position is configurable in the source code and should be adjusted according to the camera's physical entrance/exit position for a different deployment.

14. ENTRY Event

When a recognized visitor crosses the ROI line downward, an ENTRY event is generated.

The system:

Detects the ROI crossing.
Attempts to obtain the current face crop.
Saves an evidence image.
Creates an event record in SQLite.
Writes the event to events.log.

Example database event:

Visitor ID : 30
Track ID   : 52
Event      : ENTRY
Timestamp  : 2026-10-04T07:28:13
15. EXIT Event

When a recognized visitor crosses the ROI line upward, an EXIT event is generated.

The system:

Detects the ROI crossing.
Attempts to obtain the current face crop.
Saves an evidence image.
Creates an event record in SQLite.
Writes the event to events.log.

Example:

Visitor ID : 17
Track ID   : 54
Event      : EXIT
Timestamp  : 2026-10-04T07:28:52
16. Event Evidence Images

The system saves an image for every generated ENTRY and EXIT event.

ENTRY images are stored under:

logs/entries/YYYY-MM-DD/

EXIT images are stored under:

logs/exits/YYYY-MM-DD/

Example:

logs/
│
├── entries/
│   └── 2026-10-04/
│       └── visitor_30_track_52_20261004_072813_....jpg
│
└── exits/
    └── 2026-10-04/
        └── visitor_17_track_54_20261004_072852_....jpg

The corresponding image path is stored in the SQLite events table.

17. SQLite Database

The project uses SQLite for local persistent storage.

The database is:

database/visitors.db

The database contains two primary application tables.

Visitors Table
visitors
--------------------------------
visitor_id
first_seen
last_seen
visit_count
embedding
Events Table
events
--------------------------------
event_id
visitor_id
track_id
event_type
timestamp
image_path
details

The events table links events to registered visitors using visitor_id.

18. Example Database Event Records

The latest end-to-end test generated:

Visitors: 31
Events: 2

The generated events included:

ENTRY
Visitor ID : 30
Track ID   : 52
Timestamp  : 2026-10-04T07:28:13
Image      : logs/entries/2026-10-04/visitor_30_track_52_....jpg

and:

EXIT
Visitor ID : 17
Track ID   : 54
Timestamp  : 2026-10-04T07:28:52
Image      : logs/exits/2026-10-04/visitor_17_track_54_....jpg

The number of events depends on the actual movement of people across the configured ROI in the input video.

19. Event Logging

The project maintains a structured event log:

logs/events.log

Important system events include:

SYSTEM_START
TRACK_START
EMBEDDING_GENERATED
NEW_VISITOR
RECOGNIZED
ENTRY
EXIT
TRACK_LOST
SYSTEM_STOP

Example:

2026-10-04T07:29:28 | event=EMBEDDING_GENERATED | visitor_id=0 | track_id=153 | Embedding generated for face size 92x102

2026-10-04T07:29:28 | event=NEW_VISITOR | visitor_id=31 | track_id=153 | New face registered; best_similarity=0.338

2026-10-04T07:29:39 | event=RECOGNIZED | visitor_id=25 | track_id=176 | similarity=0.666

This provides a trace of recognition, registration, embedding generation, tracking, and system events.

20. Duplicate Event Protection

The system maintains the latest ENTRY/EXIT state for each visitor.

This provides basic protection against duplicate same-direction events caused by repeated crossing detections or multiple tracking IDs.

For example:

Visitor 14 -> ENTRY
Visitor 14 -> ENTRY

The second same-direction event is ignored.

A valid repeated visit sequence can be:

Visitor 14 -> ENTRY
Visitor 14 -> EXIT
Visitor 14 -> ENTRY

This allows alternating visits while reducing duplicate event generation.

21. Configuration

The main configuration is stored in:

config.json

Current configuration:

{
    "video_source": "data/sample_video.mp4",
    "detection_skip_frames": 5,
    "face_similarity_threshold": 0.60,
    "min_face_size": 80,
    "database_path": "database/visitors.db",
    "entry_log_dir": "logs/entries",
    "exit_log_dir": "logs/exits",
    "output_video": "outputs/output.mp4",
    "event_log": "logs/events.log"
}
22. Configuration Parameters
video_source

Specifies the input video or RTSP stream.

Local video example:

"video_source": "data/sample_video.mp4"

RTSP example:

"video_source": "rtsp://username:password@camera-ip:554/stream"
detection_skip_frames

Controls how frequently the detection/tracking processing is performed.

Current value:

5

This provides a configurable trade-off between processing load and detection frequency.

face_similarity_threshold

Controls the minimum cosine similarity required to recognize an existing visitor.

Current value:

0.60

A higher value makes recognition stricter and can reduce false matches, while an excessively high value may cause genuine visitors to be registered again.

min_face_size

Defines the minimum face width and height accepted for recognition.

Current value:

80 pixels

Very small face detections are ignored for direct recognition.

23. RTSP Support

The system uses OpenCV VideoCapture for its input source.

Therefore, the same processing pipeline can be used with:

Local video files.
RTSP camera streams.

Example:

{
    "video_source": "rtsp://username:password@camera-ip:554/stream"
}

Switching the input source does not require changes to the YOLO, ByteTrack, InsightFace, database, or event-processing pipeline.

For a production deployment, RTSP reconnection handling, camera health monitoring, and secure credential management should be added.

24. Installation

Create a Python virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Make sure the required model files are available before running the application.

25. Running the Project

Run the main application using:

python src\main.py

The system performs the following operations:

Initializes the SQLite database.
Initializes the event logger.
Loads YOLO11n.
Loads InsightFace.
Opens the configured input source.
Detects people.
Tracks people using ByteTrack.
Detects faces.
Generates face embeddings.
Matches faces against registered visitors.
Registers new visitors.
Determines ROI-based ENTRY/EXIT events.
Saves event evidence images.
Stores events in SQLite.
Writes structured event logs.
Generates the processed output video.
26. Output Video

The processed video is written to:

outputs/output.mp4

The output video contains:

Person bounding boxes.
Visitor IDs.
Track IDs.
Person center points.
ENTRY/EXIT ROI line.
Frame number.
Unique visitor count.

Example label:

Visitor 20 | Track 21

ROI indicator:

ENTRY / EXIT LINE
27. Sample Run Results

The latest end-to-end test successfully processed:

Total frames     : 240
Unique visitors  : 31

Generated output:

outputs/output.mp4

Event log:

logs/events.log

Entry images:

logs/entries/

Exit images:

logs/exits/

Database:

database/visitors.db

Database validation:

Visitors: 31
Events: 2

The sample run demonstrated the complete processing pipeline from video input through detection, tracking, recognition, registration, ROI crossing, event image generation, database storage, and event logging.

28. Validation and Testing

The project was validated using individual component tests before the final end-to-end run.

Tests included:

Video input validation.
YOLO detection test.
YOLO diagnostic output.
InsightFace loading test.
Face detection test.
Face embedding generation.
Face matching.
ByteTrack tracking.
SQLite initialization.
Logger initialization.
Entry/exit logic.
Full end-to-end processing.

The final pipeline successfully processed all 240 frames of the sample video.

29. Compute Load

The current implementation uses CPU inference.

YOLO11n

Purpose:

Person detection
InsightFace Buffalo_L

Purpose:

Face detection
Face embedding generation

Execution provider:

CPUExecutionProvider
ByteTrack

Purpose:

Multi-object tracking
SQLite

Purpose:

Local persistent visitor and event storage

The CPU implementation is suitable for development and demonstration using the provided sample video.

For high-resolution live RTSP streams, multiple cameras, or production-scale deployment, GPU acceleration would improve inference throughput and reduce latency.

30. Assumptions

The current implementation makes the following assumptions:

The camera provides a sufficiently clear view of visitors.
Faces should be large enough for reliable recognition.
The configured ROI line represents the logical entrance/exit boundary.
Downward crossing of the ROI is treated as ENTRY.
Upward crossing of the ROI is treated as EXIT.
The camera position is reasonably stable.
The sample video is used for development and validation.
Recognition quality depends on face size, lighting, pose, occlusion, and camera quality.
CPU inference is used for the current demonstration.
An RTSP stream can replace the sample video through config.json.
31. Limitations

The current system has the following limitations:

CPU inference is slower than GPU inference.
Very small faces may not be recognized reliably.
Heavy occlusion can affect face recognition.
Large pose changes can affect face similarity.
Challenging lighting can reduce recognition quality.
Crowded scenes can make tracking more difficult.
The ROI line must be positioned appropriately for the camera.
Face recognition depends on the quality of the available face observation.
The current implementation is primarily designed to demonstrate the required hackathon pipeline.
Production-scale multi-camera deployment would require additional infrastructure.
32. Future Improvements

Possible future improvements include:

GPU acceleration.
Multi-frame face embedding aggregation.
Stronger face re-identification.
Improved handling of occluded faces.
Multiple observations before visitor registration.
RTSP reconnection handling.
Camera health monitoring.
Multi-camera tracking.
PostgreSQL for distributed deployments.
Redis for fast temporary state.
Web dashboard.
Real-time visitor analytics.
Docker-based deployment.
Better long-term visitor re-identification.
More advanced event deduplication.
Improved visitor embedding management.
Automatic camera/ROI calibration.

33. AI-Assisted Development and Planning

AI coding assistants were used during development for:

Understanding the hackathon requirements.
Planning the project architecture.
Designing the project folder structure.
Generating modular Python code.
Debugging dependency and Python errors.
Understanding YOLO integration.
Understanding ByteTrack tracking.
Understanding InsightFace embeddings.
Designing cosine similarity matching.
Designing SQLite tables.
Designing event logging.
Designing ROI-based ENTRY/EXIT detection.
Reviewing the implementation against the hackathon requirements.
Preparing project documentation.

AI-generated code was not treated as a final black box.

The generated code was reviewed, modified, executed, debugged, and tested locally.

The development process was iterative:

Requirement Analysis
        |
        v
Project Structure
        |
        v
YOLO Detection Test
        |
        v
InsightFace Test
        |
        v
Face Matching Test
        |
        v
ByteTrack Integration
        |
        v
SQLite Integration
        |
        v
Entry/Exit Detection
        |
        v
Event Image + Database Logging
        |
        v
End-to-End Validation
        |
        v
Documentation
34. Architecture Design Decisions
Why YOLO11n?

YOLO11n provides fast person detection and is suitable for video processing applications.

Why ByteTrack?

ByteTrack maintains tracking identities across frames and provides track IDs required for movement analysis.

Why InsightFace?

InsightFace provides face detection and face embedding capabilities suitable for face recognition.

Why SQLite?

SQLite provides lightweight local persistent storage without requiring an external database server. It is suitable for the current single-camera demonstration.

Why ROI-Based Entry/Exit?

The ROI line provides a simple, explainable, and configurable method for determining directional movement in the camera view.

35. Data Flow

The complete data flow is:

Camera / Video
      |
      v
Person Detection
      |
      v
Person Tracking
      |
      v
Person Crop
      |
      v
Face Detection
      |
      v
Face Embedding
      |
      v
Face Matching
      |
      +-------------------+
      |                   |
      v                   v
Existing Visitor      New Visitor
      |                   |
      v                   v
Visitor ID            Register ID
      |                   |
      +---------+---------+
                |
                v
          ROI Movement
           /        \
        ENTRY       EXIT
          |           |
          v           v
      Save Image  Save Image
          |           |
          +-----+-----+
                |
                v
           SQLite Event
                |
                v
            Event Log
36. Error Handling and Resilience

The system performs basic validation for:

Video source availability.
Output video writer initialization.
Face detection failures.
Invalid face crops.
Small face detections.
Invalid stored embeddings.
Missing or lost tracking IDs.

Tracking information is removed after the configured missing-frame timeout.

The database and logs are stored locally so that visitor and event information can be inspected independently of the processed output video.

For production deployment, additional resilience mechanisms such as automatic RTSP reconnection, database backup, process recovery, and health monitoring can be added.

37. Security and Privacy Considerations

The system stores face embeddings and event images locally.

For production deployment:

Database access should be restricted.
Stored images should have appropriate access controls.
RTSP credentials should never be committed to GitHub.
Sensitive camera URLs should be stored using environment variables or secure configuration.
Retention policies should be defined for face images and embeddings.
Access to visitor identity data should be limited to authorized users.
Data storage and processing should follow applicable privacy requirements.

38. GitHub Repository

Repository:

https://github.com/bhavanitha-jpg/Katomaran-Hackathon

The repository contains the source code, configuration, project documentation, and required project files.

Runtime-generated files such as logs, databases, and processed output files are excluded from version control where appropriate.

39. Demo Video
  
[Watch the project demonstration on Loom](https://www.loom.com/share/2e3d57e5840445699ffa609582142282) 

The demonstration should cover:
- Project overview
- System architecture
- Live/sample video processing
- Person detection and tracking
- Face recognition and auto-registration
- Entry and exit event logging
- Evidence images
- SQLite database results

40. Submission Checklist
Core Functionality
 YOLO-based detection
 InsightFace face recognition
 ByteTrack tracking
 Automatic face registration
 Unique visitor counting
 Configurable frame skipping
 SQLite database
 ENTRY detection
 EXIT detection
 Timestamped ENTRY images
 Timestamped EXIT images
 Event database table
 Event image paths
 Event log
 Recognition logging
 Embedding generation logging
 Tracking logging
 Output video
Documentation
 Project overview
 Architecture
 Processing workflow
 Technologies
 Setup instructions
 Configuration
 RTSP explanation
 Assumptions
 Limitations
 Future improvements
 AI-assisted development planning
 Compute-load documentation
 Sample results
 Demo video link
Submission
 GitHub repository
 README
 Source code
 Configuration file
 Sample video
 Final demo video link

41. Conclusion

This project implements an end-to-end intelligent visitor tracking pipeline using modern computer vision and face recognition technologies.

The system combines:

YOLO11n
   +
ByteTrack
   +
InsightFace
   +
SQLite
   +
OpenCV
   +
ROI Entry/Exit Detection
   +
Structured Logging

The system automatically detects and tracks people, generates face embeddings, recognizes previously registered visitors, registers new faces, maintains a unique visitor count, detects directional ENTRY/EXIT events, stores event metadata in SQLite, and saves timestamped evidence images.

The implementation is designed to work with the provided sample video and can be configured for an RTSP camera stream.

The latest end-to-end validation successfully processed all 240 frames of the provided sample video and demonstrated visitor registration, recognition, tracking, event logging, database storage, and ROI-based event detection.

42. Hackathon Statement

This project is a part of a hackathon run by https://katomaran.com