# backend/database.py

import sqlite3

def init_db():
    conn = sqlite3.connect("queries.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            label TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_result(text, label, confidence):
    conn = sqlite3.connect("queries.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (text, label, confidence) VALUES (?, ?, ?)",
                   (text, label, confidence))
    conn.commit()
    conn.close()
