import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "signs.db")

SQL_CREATE = '''
CREATE TABLE IF NOT EXISTS signs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT NOT NULL,
  aliases TEXT,
  landmarks_json TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL
);
'''


def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the database and table if missing."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.executescript(SQL_CREATE)
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> Dict:
    if row is None:
        return None
    d = dict(row)
    # parse JSON fields if present
    for k in ("aliases", "landmarks_json", "metadata_json"):
        if d.get(k) is not None:
            try:
                d[k] = json.loads(d[k])
            except Exception:
                d[k] = d[k]
    return d


def get_all_signs() -> List[Dict]:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM signs ORDER BY id DESC")
        rows = cur.fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_sign(sign_id: int) -> Optional[Dict]:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM signs WHERE id = ?", (sign_id,))
        row = cur.fetchone()
        return row_to_dict(row)
    finally:
        conn.close()


def create_sign(label: str, aliases=None, landmarks=None, metadata=None) -> int:
    aliases_json = json.dumps(aliases or [])
    landmarks_json = json.dumps(landmarks or {})
    metadata_json = json.dumps(metadata or {})
    created_at = datetime.utcnow().isoformat()
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO signs (label, aliases, landmarks_json, metadata_json, created_at) VALUES (?,?,?,?,?)",
            (label, aliases_json, landmarks_json, metadata_json, created_at),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_sign(sign_id: int, label: Optional[str] = None, aliases=None, landmarks=None, metadata=None) -> bool:
    parts = []
    vals = []
    if label is not None:
        parts.append("label = ?")
        vals.append(label)
    if aliases is not None:
        parts.append("aliases = ?")
        vals.append(json.dumps(aliases))
    if landmarks is not None:
        parts.append("landmarks_json = ?")
        vals.append(json.dumps(landmarks))
    if metadata is not None:
        parts.append("metadata_json = ?")
        vals.append(json.dumps(metadata))
    if not parts:
        return False
    vals.append(sign_id)
    sql = f"UPDATE signs SET {', '.join(parts)} WHERE id = ?"
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, tuple(vals))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_sign(sign_id: int) -> bool:
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM signs WHERE id = ?", (sign_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def export_all() -> List[Dict]:
    return get_all_signs()


def import_list(signs: List[Dict], replace: bool = False) -> int:
    """Import a list of sign dicts. If replace is True, drop existing rows first.
    Returns the number of inserted rows.
    """
    if replace:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM signs")
            conn.commit()
        finally:
            conn.close()
    inserted = 0
    for s in signs:
        label = s.get("label") or s.get("name")
        aliases = s.get("aliases")
        landmarks = s.get("landmarks") or s.get("landmarks_json")
        metadata = s.get("metadata") or {}
        if not label:
            continue
        create_sign(label=label, aliases=aliases, landmarks=landmarks, metadata=metadata)
        inserted += 1
    return inserted


# Ensure DB on import
init_db()
