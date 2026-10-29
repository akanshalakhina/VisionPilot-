"""SQLite storage: the audit log, agent runs and the agent's memory."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

_RUN_JSON_FIELDS = ("plan", "grounding", "trace", "result")


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT NOT NULL,
                actor   TEXT NOT NULL,
                action  TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                goal        TEXT NOT NULL,
                budget      REAL NOT NULL,
                status      TEXT NOT NULL,
                planner     TEXT NOT NULL,
                explanation TEXT NOT NULL,
                plan        TEXT NOT NULL,
                grounding   TEXT NOT NULL,
                trace       TEXT NOT NULL,
                result      TEXT
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


# ---- audit log -----------------------------------------------------------

def log_event(db_path: str, actor: str, action: str, payload: dict) -> int:
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO audit_log (ts, actor, action, payload) VALUES (?, ?, ?, ?)",
            (_now(), actor, action, json.dumps(payload)),
        )
        return int(cur.lastrowid)


def list_events(db_path: str, limit: int = 100) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, ts, actor, action, payload FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]


# ---- agent runs ----------------------------------------------------------

def _decode_run(row: sqlite3.Row) -> dict:
    run = dict(row)
    for name in _RUN_JSON_FIELDS:
        run[name] = json.loads(run[name]) if run[name] is not None else None
    return run


def create_run(
    db_path: str, goal: str, budget: float, planner: str, explanation: str,
    plan: dict, grounding: dict, trace: list,
) -> int:
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO agent_runs (ts, goal, budget, status, planner, explanation, plan, grounding, trace)"
            " VALUES (?, ?, ?, 'proposed', ?, ?, ?, ?, ?)",
            (_now(), goal, budget, planner, explanation, json.dumps(plan), json.dumps(grounding), json.dumps(trace)),
        )
        return int(cur.lastrowid)


def get_run(db_path: str, run_id: int) -> dict | None:
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
    return _decode_run(row) if row else None


def list_runs(db_path: str, limit: int = 20) -> list[dict]:
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_decode_run(r) for r in rows]


def update_run(db_path: str, run_id: int, status: str, result: dict | None = None) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "UPDATE agent_runs SET status = ?, result = COALESCE(?, result) WHERE id = ?",
            (status, json.dumps(result) if result is not None else None, run_id),
        )


# ---- memory --------------------------------------------------------------

def memory_get(db_path: str, key: str, default=None):
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def memory_set(db_path: str, key: str, value) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO memory (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
