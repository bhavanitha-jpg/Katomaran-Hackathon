import sqlite3
import os
import json
from datetime import datetime


DATABASE_PATH = "database/visitors.db"


def get_connection():
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            visitor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            visit_count INTEGER DEFAULT 1,
            embedding TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


def add_visitor(embedding):
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    embedding_json = json.dumps(embedding.tolist())

    cursor.execute("""
        INSERT INTO visitors
        (first_seen, last_seen, visit_count, embedding)
        VALUES (?, ?, ?, ?)
    """, (now, now, 1, embedding_json))

    visitor_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return visitor_id


def get_all_visitors():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT visitor_id, first_seen, last_seen,
               visit_count, embedding
        FROM visitors
    """)

    visitors = cursor.fetchall()

    conn.close()

    return visitors


def update_visitor(visitor_id):
    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        UPDATE visitors
        SET last_seen = ?,
            visit_count = visit_count + 1
        WHERE visitor_id = ?
    """, (now, visitor_id))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_database()