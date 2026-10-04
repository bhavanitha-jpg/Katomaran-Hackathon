import os
from datetime import datetime


LOG_FILE = "logs/events.log"


def initialize_logger():
    os.makedirs("logs", exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("Katomaran Visitor Tracking Events\n")
            f.write("=" * 50 + "\n")


def log_event(event_type, visitor_id, track_id=None, details=""):

    timestamp = datetime.now().isoformat()

    message = (
        f"{timestamp} | "
        f"event={event_type} | "
        f"visitor_id={visitor_id} | "
        f"track_id={track_id} | "
        f"{details}\n"
    )

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message)

    print(message.strip())


if __name__ == "__main__":

    initialize_logger()

    log_event(
        "TEST",
        visitor_id=1,
        track_id=1,
        details="Logger test successful"
    )

    print("Logger initialized successfully.")