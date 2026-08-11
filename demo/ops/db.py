"""S4 — SQLite 스키마 (구현 내용.md의 DDL 그대로)."""
from __future__ import annotations

import sqlite3

from config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS risk_calendar(
    date TEXT, block INT, risk_prob REAL, risk_grade TEXT,
    model_type TEXT, updated_at TEXT, PRIMARY KEY(date, block));
CREATE TABLE IF NOT EXISTS farm_config(
    farm_id TEXT PRIMARY KEY, name TEXT, lat REAL, lon REAL,
    facility_type TEXT, last_manure_removal_date TEXT);
CREATE TABLE IF NOT EXISTS receptor(
    farm_id TEXT, lat REAL, lon REAL, dist_m REAL, bearing REAL,
    use_code TEXT, virtual_phone TEXT, consent_yn INT);
CREATE TABLE IF NOT EXISTS notification_log(
    id INTEGER PRIMARY KEY, farm_id TEXT, work_type TEXT,
    sent_at TEXT, message TEXT, approved_yn INT);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(DDL)
    return con


def upsert_risk(con: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: (date, block, risk_prob, risk_grade, model_type, updated_at)"""
    con.executemany(
        "INSERT INTO risk_calendar(date, block, risk_prob, risk_grade, model_type, updated_at) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(date, block) DO UPDATE SET "
        "risk_prob=excluded.risk_prob, risk_grade=excluded.risk_grade, "
        "model_type=excluded.model_type, updated_at=excluded.updated_at",
        rows,
    )
    con.commit()


def upsert_farm(con: sqlite3.Connection, farm: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO farm_config VALUES(?,?,?,?,?,?)",
        (farm["farm_id"], farm["name"], farm["lat"], farm["lon"],
         farm["facility_type"], farm["last_manure_removal_date"]),
    )
    con.commit()
