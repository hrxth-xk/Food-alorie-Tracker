# src/db.py
"""
Simple, reliable SQLite DB helper for Food-Calorie-Tracker.
Drop this file at src/db.py (overwrite existing).
Provides: init_db(), add_meal(...), get_logs_for_date(date_obj), get_logs_between(start_date, end_date)
"""

import os
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Optional, Any

# Make DB path absolute relative to this file (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "food_logger.db")

# Schema version note — keep minimal and easy to extend
CREATE_MEALS_TABLE = """
CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_date TEXT NOT NULL,
    food_name TEXT NOT NULL,
    calories REAL DEFAULT 0,
    protein REAL DEFAULT 0,
    carbs REAL DEFAULT 0,
    fat REAL DEFAULT 0,
    serving TEXT,
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

def _get_conn():
    """Return a new sqlite3 connection (one per call, simple and safe)."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the DB and create tables if they don't exist."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(CREATE_MEALS_TABLE)
        conn.commit()
    finally:
        conn.close()

def add_meal(
    food_name: str,
    calories: float = 0.0,
    protein: Optional[float] = None,
    carbs: Optional[float] = None,
    fat: Optional[float] = None,
    serving: Optional[str] = None,
    source: Optional[str] = None,
    log_date: Optional[date] = None,
) -> int:
    """
    Insert a meal log and return the inserted row id.
    - log_date: date object or ISO string. Defaults to today.
    """
    if log_date is None:
        log_date = date.today()
    if isinstance(log_date, datetime):
        log_date = log_date.date()
    # Ensure numeric values are floats (or NULL)
    def _num(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    calories_v = _num(calories)
    protein_v = _num(protein)
    carbs_v = _num(carbs)
    fat_v = _num(fat)

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO meals (log_date, food_name, calories, protein, carbs, fat, serving, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_date.isoformat(),
                str(food_name),
                calories_v,
                protein_v,
                carbs_v,
                fat_v,
                serving,
                source,
            ),
        )
        rowid = cur.lastrowid
        conn.commit()
        return rowid
    finally:
        conn.close()

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}

def get_logs_for_date(log_date: date) -> List[Dict[str, Any]]:
    """Return list of rows (dicts) for the given date (date object or ISO string)."""
    if isinstance(log_date, datetime):
        log_date = log_date.date()
    if isinstance(log_date, str):
        date_str = log_date
    else:
        date_str = log_date.isoformat()

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM meals WHERE log_date = ? ORDER BY created_at ASC, id ASC", (date_str,))
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()

def get_logs_between(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """
    Return rows between start_date and end_date (inclusive).
    Accepts date objects or ISO strings.
    """
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()
    s = start_date.isoformat() if not isinstance(start_date, str) else start_date
    e = end_date.isoformat() if not isinstance(end_date, str) else end_date

    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM meals WHERE log_date BETWEEN ? AND ? ORDER BY log_date ASC, created_at ASC, id ASC",
            (s, e),
        )
        rows = cur.fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()

# simple convenience for debugging
if __name__ == "__main__":
    print("DB_PATH =", DB_PATH)
    init_db()
    print("initialized")
