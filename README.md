\# Katomaran Visitor Tracking System



\## 1. Project Overview



Katomaran Visitor Tracking System is an AI-based computer vision system designed to detect, track, recognize, and monitor visitors from surveillance video.



The system combines:



\- YOLO11 for person detection

\- ByteTrack for multi-person tracking

\- InsightFace for face recognition

\- SQLite for visitor identity storage

\- ROI-based movement detection for Entry/Exit events

\- Event logging for maintaining visitor activity history



\---



\## 2. System Workflow



Video Input

&#x20;   ↓

YOLO11 Person Detection

&#x20;   ↓

ByteTrack Multi-Person Tracking

&#x20;   ↓

Face Detection \& Recognition

&#x20;   ↓

Visitor Identification

&#x20;   ↓

ROI Line Crossing Detection

&#x20;   ↓

Entry / Exit Event

&#x20;   ↓

SQLite Database + Event Logs



\---



\## 3. Technologies Used



| Technology | Purpose |

|------------|---------|

| Python | Main programming language |

| OpenCV | Video processing and visualization |

| YOLO11 | Person detection |

| ByteTrack | Multi-object tracking |

| InsightFace | Face detection and recognition |

| NumPy | Numerical processing |

| SQLite | Visitor database |

| JSON | Configuration |

| PowerShell | Project execution |



\---



\## 4. Project Structure



```text

Katomaran\_Hackathon/

│

├── data/

│   └── sample\_video.mp4

│

├── database/

│   └── visitors.db

│

├── logs/

│   ├── entries/

│   ├── exits/

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

│   ├── yolo\_test.py

│   └── tracking\_test.py

│

├── config.json

├── requirements.txt

└── README.md

