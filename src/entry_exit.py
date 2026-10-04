import os
from datetime import datetime

from logger import log_event


ENTRY_DIR = "logs/entries"
EXIT_DIR = "logs/exits"

EXIT_TIMEOUT_FRAMES = 30


class EntryExitManager:

    def __init__(self):
        os.makedirs(ENTRY_DIR, exist_ok=True)
        os.makedirs(EXIT_DIR, exist_ok=True)

        self.active_visitors = {}

    def visitor_entered(self, visitor_id, track_id, frame=None):

        if track_id in self.active_visitors:
            return

        self.active_visitors[track_id] = {
            "visitor_id": visitor_id,
            "last_seen_frame": 0
        }

        log_event(
            "ENTRY",
            visitor_id=visitor_id,
            track_id=track_id,
            details="Visitor entered"
        )

    def update_seen(self, track_id, frame_number):

        if track_id in self.active_visitors:
            self.active_visitors[track_id]["last_seen_frame"] = frame_number

    def check_exits(self, current_frame):

        exited = []

        for track_id, data in list(
            self.active_visitors.items()
        ):

            last_seen = data["last_seen_frame"]

            if (
                current_frame - last_seen
                >= EXIT_TIMEOUT_FRAMES
            ):

                visitor_id = data["visitor_id"]

                log_event(
                    "EXIT",
                    visitor_id=visitor_id,
                    track_id=track_id,
                    details="Visitor exited"
                )

                exited.append(track_id)

        for track_id in exited:
            del self.active_visitors[track_id]

        return exited

    def get_active_visitors(self):

        return self.active_visitors


if __name__ == "__main__":

    manager = EntryExitManager()

    manager.visitor_entered(
        visitor_id=1,
        track_id=10
    )

    manager.update_seen(
        track_id=10,
        frame_number=1
    )

    print(
        "Active visitors:",
        manager.get_active_visitors()
    )

    print("Entry/Exit manager test completed.")