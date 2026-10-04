import sqlite3
import os
import json
from datetime import datetime


DATABASE_PATH = "database/visitors.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """
    Create all required database tables if they do not already exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # Visitors table
    # ---------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            visitor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            visit_count INTEGER DEFAULT 1,
            embedding TEXT NOT NULL
        )
    """)

    # ---------------------------------------------------------
    # Events table
    # ---------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id INTEGER NOT NULL,
            track_id INTEGER,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            details TEXT,
            FOREIGN KEY (visitor_id)
                REFERENCES visitors(visitor_id)
        )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


def add_visitor(embedding):
    """
    Register a new visitor and store their face embedding.
    """

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    embedding_json = json.dumps(
        embedding.tolist()
    )

    cursor.execute("""
        INSERT INTO visitors
        (first_seen, last_seen, visit_count, embedding)
        VALUES (?, ?, ?, ?)
    """, (
        now,
        now,
        1,
        embedding_json
    ))

    visitor_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return visitor_id


def get_all_visitors():
    """
    Retrieve all registered visitors.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            visitor_id,
            first_seen,
            last_seen,
            visit_count,
            embedding
        FROM visitors
    """)

    visitors = cursor.fetchall()

    conn.close()

    return visitors


def update_visitor(visitor_id):
    """
    Update the last-seen timestamp for a visitor.
    """

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        UPDATE visitors
        SET last_seen = ?,
            visit_count = visit_count + 1
        WHERE visitor_id = ?
    """, (
        now,
        visitor_id
    ))

    conn.commit()
    conn.close()


def add_event(
    visitor_id,
    track_id,
    event_type,
    timestamp,
    image_path=None,
    details=""
):
    """
    Store an ENTRY or EXIT event in the database.

    Parameters:
        visitor_id  : Persistent visitor identity
        track_id    : Temporary tracker identity
        event_type  : ENTRY / EXIT
        timestamp   : Event timestamp
        image_path  : Path to saved event image
        details     : Additional event information
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events
        (
            visitor_id,
            track_id,
            event_type,
            timestamp,
            image_path,
            details
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        visitor_id,
        track_id,
        event_type,
        timestamp,
        image_path,
        details
    ))

    event_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return event_id


def get_all_events():
    """
    Retrieve all recorded events.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            event_id,
            visitor_id,
            track_id,
            event_type,
            timestamp,
            image_path,
            details
        FROM events
        ORDER BY timestamp
    """)

    events = cursor.fetchall()

    conn.close()

    return events


def get_event_count():
    """
    Return the total number of recorded events.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM events
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


def get_unique_visitor_count():
    """
    Return the number of unique registered visitors.
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM visitors
    """)

    count = cursor.fetchone()[0]

    conn.close()

    return count


if __name__ == "__main__":

    initialize_database()

    print(
        "Unique visitors:",
        get_unique_visitor_count()
    )

    print(
        "Total events:",
        get_event_count()
    )