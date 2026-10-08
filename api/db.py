"""
3DINVENZA — SQLite Model Database
Persists every generated 3D model so the gallery can browse/reopen them.
"""

import sqlite3
import os
import json
import time
import hashlib
import secrets
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "models.db"

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return hashed.hex(), salt

def verify_password(password: str, hashed: str, salt: str) -> bool:
    new_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(new_hash, hashed)


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _get_conn():
    """Thread-safe connection context manager."""
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Called once on app startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT    UNIQUE NOT NULL,
                name            TEXT    NOT NULL DEFAULT 'Untitled Model',
                engine          TEXT    DEFAULT '',
                engine_label    TEXT    DEFAULT '',
                mode            TEXT    DEFAULT 'single_image',
                object_type     TEXT    DEFAULT 'organic',
                quality         TEXT    DEFAULT 'high',
                status          TEXT    DEFAULT 'completed',
                created_at      REAL    NOT NULL,
                thumbnail_path  TEXT    DEFAULT '',
                stl_path        TEXT    DEFAULT '',
                glb_path        TEXT    DEFAULT '',
                ply_path        TEXT    DEFAULT '',
                vertices        INTEGER DEFAULT 0,
                faces           INTEGER DEFAULT 0,
                length_mm       REAL    DEFAULT 0,
                width_mm        REAL    DEFAULT 0,
                height_mm       REAL    DEFAULT 0,
                volume_mm3      REAL    DEFAULT 0,
                is_watertight   INTEGER DEFAULT 0,
                file_size_bytes INTEGER DEFAULT 0,
                extra_json      TEXT    DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_models_created
            ON models(created_at DESC)
        """)
        
        # Auth Tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    UNIQUE NOT NULL,
                email           TEXT    UNIQUE NOT NULL,
                password_hash   TEXT    NOT NULL,
                salt            TEXT    NOT NULL,
                created_at      REAL    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id         INTEGER PRIMARY KEY,
                theme           TEXT    DEFAULT 'dark',
                default_engine  TEXT    DEFAULT 'trellis',
                notifications   INTEGER DEFAULT 1,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Migration: Add is_admin column if it doesn't exist
        try:
            conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Column already exists
            
    print(f"[DB] Model database initialized at {DB_PATH}")


def save_model(
    job_id: str,
    name: str = "",
    engine: str = "",
    engine_label: str = "",
    mode: str = "single_image",
    object_type: str = "organic",
    quality: str = "high",
    stl_path: str = "",
    glb_path: str = "",
    ply_path: str = "",
    vertices: int = 0,
    faces: int = 0,
    length_mm: float = 0,
    width_mm: float = 0,
    height_mm: float = 0,
    volume_mm3: float = 0,
    is_watertight: bool = False,
    file_size_bytes: int = 0,
    thumbnail_path: str = "",
    extra: Optional[Dict] = None,
) -> int:
    """Insert or update a model record. Returns the row id."""
    if not name:
        name = f"Scan {job_id}"

    # Calculate file size if not provided
    if file_size_bytes == 0:
        for p in [glb_path, stl_path]:
            if p and os.path.isfile(p):
                file_size_bytes = os.path.getsize(p)
                break

    with _get_conn() as conn:
        # Upsert: update if job_id already exists
        existing = conn.execute(
            "SELECT id FROM models WHERE job_id = ?", (job_id,)
        ).fetchone()

        if existing:
            conn.execute("""
                UPDATE models SET
                    name=?, engine=?, engine_label=?, mode=?, object_type=?,
                    quality=?, status='completed', stl_path=?, glb_path=?,
                    ply_path=?, vertices=?, faces=?, length_mm=?, width_mm=?,
                    height_mm=?, volume_mm3=?, is_watertight=?,
                    file_size_bytes=?, thumbnail_path=?, extra_json=?
                WHERE job_id=?
            """, (
                name, engine, engine_label, mode, object_type, quality,
                stl_path, glb_path, ply_path, vertices, faces,
                length_mm, width_mm, height_mm, volume_mm3,
                int(is_watertight), file_size_bytes, thumbnail_path,
                json.dumps(extra or {}), job_id,
            ))
            return existing["id"]
        else:
            cursor = conn.execute("""
                INSERT INTO models (
                    job_id, name, engine, engine_label, mode, object_type,
                    quality, status, created_at, stl_path, glb_path, ply_path,
                    vertices, faces, length_mm, width_mm, height_mm,
                    volume_mm3, is_watertight, file_size_bytes,
                    thumbnail_path, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, name, engine, engine_label, mode, object_type,
                quality, time.time(), stl_path, glb_path, ply_path,
                vertices, faces, length_mm, width_mm, height_mm,
                volume_mm3, int(is_watertight), file_size_bytes,
                thumbnail_path, json.dumps(extra or {}),
            ))
            return cursor.lastrowid


def get_all_models(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Return all models ordered by newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def get_model(model_id: int) -> Optional[Dict]:
    """Return a single model by its primary key id."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        return dict(row) if row else None


def get_model_by_job(job_id: str) -> Optional[Dict]:
    """Return a single model by job_id."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE job_id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_model(model_id: int) -> bool:
    """Delete a model record. Returns True if a row was deleted."""
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM models WHERE id = ?", (model_id,)
        )
        return cursor.rowcount > 0


def get_model_count() -> int:
    """Return total number of saved models."""
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM models").fetchone()
        return row["cnt"] if row else 0

# ==============================================================================
# AUTH & SETTINGS
# ==============================================================================

def create_user(username: str, email: str, password: str, is_admin: int = 0) -> Dict:
    pwd_hash, salt = hash_password(password)
    
    # Auto-grant admin to the exact username "admin"
    if username == "admin":
        is_admin = 1
        
    with _get_conn() as conn:
        try:
            cursor = conn.execute("""
                INSERT INTO users (username, email, password_hash, salt, created_at, is_admin)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, email, pwd_hash, salt, time.time(), is_admin))
            user_id = cursor.lastrowid
            conn.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
            return {"id": user_id, "username": username, "email": email, "is_admin": is_admin}
        except sqlite3.IntegrityError:
            return None # Username or email exists

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        if verify_password(password, row["password_hash"], row["salt"]):
            return dict(row)
        return None

def get_user_settings(user_id: int) -> Dict:
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else {}

def update_user_settings(user_id: int, theme: str, default_engine: str, notifications: int) -> bool:
    with _get_conn() as conn:
        cursor = conn.execute("""
            UPDATE user_settings 
            SET theme=?, default_engine=?, notifications=? 
            WHERE user_id=?
        """, (theme, default_engine, notifications, user_id))
        return cursor.rowcount > 0

def get_all_users_db() -> List[Dict]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT id, username, email, is_admin, created_at FROM users").fetchall()
        return [dict(r) for r in rows]
