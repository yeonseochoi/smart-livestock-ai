"""S4 — SQLite 스키마 (구현 내용.md의 DDL 그대로)."""
from __future__ import annotations

import sqlite3

from config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS risk_calendar(
    date TEXT, block INT, risk_prob REAL, risk_grade TEXT,
    model_type TEXT, updated_at TEXT, PRIMARY KEY(date, block));
CREATE TABLE IF NOT EXISTS risk_calendar_v6(
    date TEXT, hour INT, grp TEXT, risk_prob REAL, risk_grade TEXT,
    model_type TEXT, updated_at TEXT,
    PRIMARY KEY(date, hour, grp));
CREATE TABLE IF NOT EXISTS forecast_hourly_v6(
    date TEXT, hour INT, wd REAL, ws REAL, sky TEXT, temp REAL, humid REAL,
    rain INT, source TEXT, updated_at TEXT,
    PRIMARY KEY(date, hour));
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


def upsert_risk_v6(con: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: (date, hour, grp, risk_prob, risk_grade, model_type, updated_at)

    ★ 기존 risk_calendar 는 PRIMARY KEY(date, block) 이라 그룹을 도입하면
      뒤 행이 앞 행을 조용히 덮어쓴다. 그래서 별도 테이블을 쓴다.
      grp 인 이유 — group 은 SQL 예약어라 컬럼명으로 못 쓴다.
    """
    con.executemany(
        "INSERT INTO risk_calendar_v6(date, hour, grp, risk_prob, risk_grade, "
        "model_type, updated_at) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(date, hour, grp) DO UPDATE SET "
        "risk_prob=excluded.risk_prob, risk_grade=excluded.risk_grade, "
        "model_type=excluded.model_type, updated_at=excluded.updated_at",
        rows,
    )
    con.commit()


def upsert_forecast_v6(con: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: (date, hour, wd, ws, sky, temp, humid, rain, source, updated_at)

    s5 의 플룸 그룹 선택이 시각별 바람을 필요로 한다. risk_calendar_v6 에는
    확률만 있으므로 예보 원값을 따로 남긴다 (재호출 방지).
    """
    con.executemany(
        "INSERT INTO forecast_hourly_v6(date, hour, wd, ws, sky, temp, humid, "
        "rain, source, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(date, hour) DO UPDATE SET "
        "wd=excluded.wd, ws=excluded.ws, sky=excluded.sky, temp=excluded.temp, "
        "humid=excluded.humid, rain=excluded.rain, source=excluded.source, "
        "updated_at=excluded.updated_at",
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
