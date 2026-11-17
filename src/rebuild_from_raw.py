import calendar
import gzip
import json
import pathlib
import sqlite3
import time

import collector

OLD_DB = "data/bazaar.db"
NEW_DB = "data/bazaar_rebuilt.db"
RAW_DIR = pathlib.Path("data/raw")


def ts_from_path(path):
    day = path.parent.name
    hhmm = path.name.split("_")[1][:4]
    return calendar.timegm(time.strptime(f"{day} {hhmm}", "%Y-%m-%d %H%M"))


def rebuild(old_db=OLD_DB, new_db=NEW_DB, raw_dir=RAW_DIR):
    new_path = pathlib.Path(new_db)
    if new_path.exists():
        new_path.unlink()
    conn = sqlite3.connect(new_db)
    conn.executescript(collector.SCHEMA)

    files = sorted(pathlib.Path(raw_dir).glob("*/bazaar_*.json.gz"))
    for path in files:
        with gzip.open(path, "rt") as f:
            payload = json.load(f)
        rows = collector.rows_from_payload(payload, ts_from_path(path))
        conn.executemany(
            "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )

    if pathlib.Path(old_db).exists():
        conn.execute("ATTACH DATABASE ? AS old", (old_db,))
        conn.execute("INSERT OR REPLACE INTO election SELECT * FROM old.election")
    conn.commit()

    snaps = conn.execute("SELECT COUNT(DISTINCT ts) FROM snapshots").fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    elections = conn.execute("SELECT COUNT(*) FROM election").fetchone()[0]
    conn.close()
    print(f"read {len(files)} raw files -> {snaps} snapshots, {rows} rows, {elections} election rows")


if __name__ == "__main__":
    rebuild()