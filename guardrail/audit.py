"""
Append-only audit log backed by SQLite.

Records every scan performed by the guardrail system — what was scanned,
when, what risk was found, and what the recommendation was. Designed for
compliance querying: "show me all high-risk events from the last 24 hours."

Design decisions:
- Append-only: only log_scan() writes; no update/delete exposed anywhere.
  This isn't hash-chained immutability (see stretch goals), but it means
  no code path in the application can silently alter past records.
- Text preview only: stores first 100 chars, never full text. If the
  scanned text contained secrets or PII, they shouldn't sit in a log file.
- Reasons stored as JSON: flexible, queryable, and human-readable.
- SQLite: zero-config, file-based, perfect for a single-server MCP tool.
  In production you'd swap this for Postgres with row-level security.
"""

from __future__ import annotations

import json
import sqlite3
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field


# ─── Data Structures ────────────────────────────────────────────────


@dataclass
class AuditEntry:
    """One audit log record, matching the SQLite table schema."""
    id: int
    timestamp: str
    scan_type: str          # "injection" or "pii"
    source: str             # Who/what triggered the scan (e.g. "user_input", "agent_output")
    risk_level: str         # "low", "medium", "high"
    risk_score: int         # 0-100
    reasons: list[str]      # Human-readable reasons from the detector
    recommendation: str     # "Proceed", "Flag for review", "Block"
    text_preview: str       # First 100 chars of the scanned text


@dataclass
class AuditStats:
    """Aggregate statistics from the audit log."""
    total_scans: int = 0
    by_risk_level: dict = field(default_factory=dict)   # {"low": 5, "medium": 3, "high": 1}
    by_scan_type: dict = field(default_factory=dict)    # {"injection": 6, "pii": 3}
    by_recommendation: dict = field(default_factory=dict)  # {"Proceed": 4, "Block": 2, ...}


# ─── Database Setup ─────────────────────────────────────────────────

def _default_db_path() -> str:
    """
    Compute the default audit database location.

    When this package is pip-installed, audit.py lives inside
    site-packages/, where writing a database file is the wrong move —
    it may not even be writable, and it pollutes the package directory.

    Instead, default to a per-user data directory:
      - Respects $AGENT_GUARDRAIL_DATA_DIR if set (useful for Docker/CI)
      - Otherwise uses $XDG_DATA_HOME or ~/.local/share on Linux/macOS,
        and %APPDATA% on Windows — the conventional "where does this
        app store its data" location on each platform.
    """
    override = os.environ.get("AGENT_GUARDRAIL_DATA_DIR")
    if override:
        base = override
    elif os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        base = os.path.join(base, "agent-guardrail-mcp")
    else:
        xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        base = os.path.join(xdg, "agent-guardrail-mcp")

    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "audit.db")


# Computed once at import time; callers can still override per-call via
# the db_path parameter on every function below (used heavily in tests).
DEFAULT_DB_PATH = _default_db_path()

# Preview length — how many characters of the scanned text to store
PREVIEW_LENGTH = 100


def _get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """
    Get a SQLite connection with row_factory set for dict-like access.

    Using sqlite3.Row as row_factory lets us access columns by name
    (row["risk_level"]) instead of by index (row[4]), which is less
    error-prone and more readable.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> None:
    """
    Create the audit_log table if it doesn't exist.

    Called automatically by log_scan() on first use, but can also be
    called explicitly during setup or testing.

    Table columns:
    - id: auto-incrementing primary key
    - timestamp: ISO-8601 UTC timestamp
    - scan_type: what kind of scan ("injection" or "pii")
    - source: where the scan request came from
    - risk_level: the detector's classification
    - risk_score: numeric score (0-100)
    - reasons_json: JSON array of human-readable matched reasons
    - recommendation: what the system recommended doing
    - text_preview: first N chars of the scanned text (never full text)
    """
    conn = _get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                scan_type       TEXT NOT NULL,
                source          TEXT NOT NULL,
                risk_level      TEXT NOT NULL,
                risk_score      INTEGER NOT NULL,
                reasons_json    TEXT NOT NULL,
                recommendation  TEXT NOT NULL,
                text_preview    TEXT NOT NULL
            )
        """)
        # Index on risk_level for filtered queries — the most common
        # audit question is "show me everything flagged as high"
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_level
            ON audit_log(risk_level)
        """)
        # Index on timestamp for time-range queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON audit_log(timestamp)
        """)
        conn.commit()
    finally:
        conn.close()


# ─── Write Operations (append-only) ─────────────────────────────────


def log_scan(
    scan_type: str,
    source: str,
    risk_level: str,
    risk_score: int,
    reasons: list[str],
    recommendation: str,
    text: str,
    db_path: str | None = None,
) -> int:
    """
    Record a scan result in the audit log.

    This is the ONLY write operation exposed by this module.
    No update, no delete — append-only by design.

    Args:
        scan_type: "injection" or "pii"
        source: Origin of the scan (e.g. "user_input", "agent_output")
        risk_level: "low", "medium", or "high"
        risk_score: Numeric score 0-100
        reasons: List of human-readable reasons from the detector
        recommendation: "Proceed", "Flag for review", or "Block"
        text: The full scanned text (only a preview is stored)
        db_path: Optional override for the database file path

    Returns:
        The id of the newly inserted audit log entry.
    """
    # Ensure the table exists
    init_db(db_path)

    # Truncate text to preview length — never store full content
    preview = text[:PREVIEW_LENGTH]
    if len(text) > PREVIEW_LENGTH:
        preview += "..."

    timestamp = datetime.now(timezone.utc).isoformat()
    reasons_json = json.dumps(reasons)

    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO audit_log
                (timestamp, scan_type, source, risk_level, risk_score,
                 reasons_json, recommendation, text_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, scan_type, source, risk_level, risk_score,
             reasons_json, recommendation, preview)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


# ─── Read Operations ────────────────────────────────────────────────


def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
    """Convert a database row to an AuditEntry dataclass."""
    return AuditEntry(
        id=row["id"],
        timestamp=row["timestamp"],
        scan_type=row["scan_type"],
        source=row["source"],
        risk_level=row["risk_level"],
        risk_score=row["risk_score"],
        reasons=json.loads(row["reasons_json"]),
        recommendation=row["recommendation"],
        text_preview=row["text_preview"],
    )


def get_audit_trail(
    limit: int = 20,
    risk_level: str | None = None,
    db_path: str | None = None,
) -> list[AuditEntry]:
    """
    Retrieve recent audit log entries, optionally filtered by risk level.

    Args:
        limit: Maximum number of entries to return (most recent first)
        risk_level: If provided, only return entries with this risk level
        db_path: Optional override for the database file path

    Returns:
        List of AuditEntry objects, most recent first.
    """
    init_db(db_path)

    conn = _get_connection(db_path)
    try:
        if risk_level:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE risk_level = ? ORDER BY id DESC LIMIT ?",
                (risk_level, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()

        return [_row_to_entry(row) for row in rows]
    finally:
        conn.close()


def get_stats(db_path: str | None = None) -> AuditStats:
    """
    Get aggregate statistics from the audit log.

    Returns counts broken down by risk level, scan type, and recommendation.
    This is the "dashboard" query — gives a quick overview of what the
    guardrail system has been doing.
    """
    init_db(db_path)

    conn = _get_connection(db_path)
    try:
        # Total count
        total = conn.execute("SELECT COUNT(*) as cnt FROM audit_log").fetchone()["cnt"]

        # Breakdown by risk level
        by_risk = {}
        for row in conn.execute(
            "SELECT risk_level, COUNT(*) as cnt FROM audit_log GROUP BY risk_level"
        ).fetchall():
            by_risk[row["risk_level"]] = row["cnt"]

        # Breakdown by scan type
        by_type = {}
        for row in conn.execute(
            "SELECT scan_type, COUNT(*) as cnt FROM audit_log GROUP BY scan_type"
        ).fetchall():
            by_type[row["scan_type"]] = row["cnt"]

        # Breakdown by recommendation
        by_rec = {}
        for row in conn.execute(
            "SELECT recommendation, COUNT(*) as cnt FROM audit_log GROUP BY recommendation"
        ).fetchall():
            by_rec[row["recommendation"]] = row["cnt"]

        return AuditStats(
            total_scans=total,
            by_risk_level=by_risk,
            by_scan_type=by_type,
            by_recommendation=by_rec,
        )
    finally:
        conn.close()