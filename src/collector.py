"""
Hypixel SkyBlock bazaar collector.

Every 15 minutes it:
  1. downloads the bazaar JSON (no API key needed)
  2. saves the untouched raw JSON (gzipped) so you can re-parse later
  3. flattens it into one row per item in SQLite (table: snapshots)
  4. saves the current mayor/election JSON (table: election)

Run:   pip install requests
       python bazaar_collector.py
Leave it running on a machine that stays on (or run it from cron / Task Scheduler
with RUN_ONCE=1 python bazaar_collector.py).
"""
import gzip
import json
import pathlib
import sqlite3
import time

import requests

BAZAAR_URL = "https://api.hypixel.net/v2/skyblock/bazaar"
ELECTION_URL = "https://api.hypixel.net/v2/resources/skyblock/election"
INTERVAL = 15 * 60
DATA_DIR = pathlib.Path("data")
DB_PATH = DATA_DIR / "bazaar.db"
RAW_DIR = DATA_DIR / "raw"
KEEP_RAW = True

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts               INTEGER NOT NULL,
    product_id       TEXT    NOT NULL,
    api_updated      INTEGER,
    instabuy_price   REAL,
    instasell_price  REAL,
    top_ask          REAL,
    top_bid          REAL,
    ask_depth_coins  REAL,
    bid_depth_coins  REAL,
    buy_volume       INTEGER,
    sell_volume      INTEGER,
    buy_orders       INTEGER,
    sell_orders      INTEGER,
    buy_moving_week  INTEGER,
    sell_moving_week INTEGER,
    PRIMARY KEY (ts, product_id)
);
CREATE TABLE IF NOT EXISTS election (
    ts      INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snap_product ON snapshots(product_id, ts);
"""


def fetch_json(url, tries=3):
    last_err = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "bazaar-research/0.1"})
            r.raise_for_status()
            data = r.json()
            if data.get("success", True):
                return data
            last_err = f"API said success=false: {data}"
        except Exception as e:
            last_err = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def levels(summary):
    prices = [lvl["pricePerUnit"] for lvl in summary or []]
    coins = sum(lvl["amount"] * lvl["pricePerUnit"] for lvl in summary or [])
    return prices, coins


def rows_from_payload(payload, ts):
    rows = []
    api_updated = payload.get("lastUpdated")
    for pid, prod in payload.get("products", {}).items():
        qs = prod.get("quick_status", {})
        ask_prices, ask_coins = levels(prod.get("buy_summary"))
        bid_prices, bid_coins = levels(prod.get("sell_summary"))
        rows.append((
            ts, pid, api_updated,
            qs.get("buyPrice"), qs.get("sellPrice"),
            min(ask_prices) if ask_prices else None,
            max(bid_prices) if bid_prices else None,
            ask_coins if ask_prices else None,
            bid_coins if bid_prices else None,
            qs.get("buyVolume"), qs.get("sellVolume"),
            qs.get("buyOrders"), qs.get("sellOrders"),
            qs.get("buyMovingWeek"), qs.get("sellMovingWeek"),
        ))
    return rows


def save_raw(name, payload, ts):
    if not KEEP_RAW:
        return
    day = time.strftime("%Y-%m-%d", time.gmtime(ts))
    folder = RAW_DIR / day
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{time.strftime('%H%M', time.gmtime(ts))}.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def snapshot_once(conn, ts):
    bazaar = fetch_json(BAZAAR_URL)
    save_raw("bazaar", bazaar, ts)
    rows = rows_from_payload(bazaar, ts)
    conn.executemany(
        "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    if ts % 3600 == 0:
        try:
            election = fetch_json(ELECTION_URL)
            save_raw("election", election, ts)
            conn.execute("INSERT OR REPLACE INTO election VALUES (?,?)", (ts, json.dumps(election)))
        except Exception as e:
            print(f"election fetch failed: {e}")
    conn.commit()
    print(f"[{time.strftime('%H:%M:%S')}] saved {len(rows)} products")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    while True:
        time.sleep(INTERVAL - time.time() % INTERVAL + 1)
        ts = int(time.time() // INTERVAL * INTERVAL)
        try:
            snapshot_once(conn, ts)
        except Exception as e:
            print(f"snapshot failed, will retry next cycle: {e}")


if __name__ == "__main__":
    main()