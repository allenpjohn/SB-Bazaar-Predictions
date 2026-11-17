import json
import sqlite3
import os

import numpy as np
import pandas as pd

DB_PATH = "data/bazaar.db"
TIMEZONE = "America/Chicago"

FIRST_TERM_START = "2026-11-15 17:15"

ITEM_CATEGORIES_CSV = "reference/item_categories.csv"
EVENTS_CSV = "reference/events.csv"
PERKS_CSV = "reference/perks.csv"

ORDER_SIZE_COINS = 10_000_000
MAX_FILL_HOURS = 0.5
FILL_WEIGHT_CAP = 5

MIN_TURNOVER_PER_HR = 100_000_000
OUTLIER_RATIO = 3
SNAPS_PER_HOUR = 4
HOURS_PER_WEEK = 168


def load_snapshots(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM snapshots", conn)
    conn.close()
    return df


def clean(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    for col in ["instabuy_price", "instasell_price"]:
        df[col] = df[col].replace(0, np.nan)
    return df.sort_values("ts").reset_index(drop=True)


def load_mayor(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    rows = []
    for ts, payload in conn.execute("SELECT ts, payload FROM election ORDER BY ts"):
        m = json.loads(payload).get("mayor") or {}
        minister = m.get("minister") or {}
        perk = minister.get("perk") or {}
        rows.append({
            "ts": ts,
            "mayor": m.get("name"),
            "mayor_perks": "|".join(p.get("name", "") for p in m.get("perks") or []),
            "minister": minister.get("name"),
            "minister_perk": perk.get("name"),
        })
    conn.close()

    cols = ["ts", "mayor", "mayor_perks", "minister", "minister_perk", "term_start"]
    if not rows:
        return pd.DataFrame(columns=cols)

    mayor = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    key = mayor["mayor"].fillna("") + "|" + mayor["minister"].fillna("")
    changed = key != key.shift()
    mayor["term_start"] = mayor["ts"].where(changed).ffill()

    first_term = mayor["term_start"] == mayor["ts"].iloc[0]
    if FIRST_TERM_START:
        override = pd.Timestamp(FIRST_TERM_START, tz=TIMEZONE).timestamp()
        mayor.loc[first_term, "term_start"] = override
    else:
        mayor.loc[first_term, "term_start"] = np.nan
    return mayor[cols]


def add_mayor_features(df, mayor):
    if mayor.empty:
        for col in ["mayor", "mayor_perks", "minister", "minister_perk", "term_start"]:
            df[col] = np.nan
    else:
        df = pd.merge_asof(df.sort_values("ts"), mayor, on="ts", direction="backward")
    df["hours_since_office"] = (df["ts"] - df["term_start"]) / 3600
    for col in ["mayor", "minister", "minister_perk"]:
        df[col] = df[col].astype("category")
    return df


def add_market_features(df):
    df = df.sort_values(["product_id", "ts"]).reset_index(drop=True)

    df["mid"] = (df["top_bid"] + df["top_ask"]) / 2
    df["log_mid"] = np.log(df["mid"])
    df["spread_pct"] = (df["top_ask"] - df["top_bid"]) / df["mid"]

    df["sell_per_hr"] = df["sell_moving_week"] / HOURS_PER_WEEK
    df["buy_per_hr"] = df["buy_moving_week"] / HOURS_PER_WEEK
    df["turnover_per_hr"] = df["instasell_price"] * df["sell_per_hr"]
    df["buy_turnover_per_hr"] = df["instabuy_price"] * df["buy_per_hr"]
    df["two_sided_turnover"] = np.minimum(df["turnover_per_hr"], df["buy_turnover_per_hr"])
    df["est_fill_hours"] = ORDER_SIZE_COINS / df["two_sided_turnover"].replace(0, np.nan)
    df["fill_weight"] = (df["two_sided_turnover"] / MIN_TURNOVER_PER_HR).clip(0, FILL_WEIGHT_CAP)

    total_volume = (df["buy_volume"] + df["sell_volume"]).replace(0, np.nan)
    df["book_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / total_volume
    depth = (df["bid_depth_coins"] / df["ask_depth_coins"]).replace([np.inf, -np.inf], np.nan)
    df["depth_ratio_log"] = np.log(depth.where(depth > 0))

    g = df.groupby("product_id", sort=False)
    for name, n in [("1h", 1), ("6h", 6), ("24h", 24), ("7d", 168)]:
        df[f"ret_{name}"] = df["log_mid"] - g["log_mid"].shift(n * SNAPS_PER_HOUR)

    df["r1"] = g["log_mid"].diff()
    g = df.groupby("product_id", sort=False)
    day = 24 * SNAPS_PER_HOUR
    df["vol_24h"] = g["r1"].transform(lambda s: s.rolling(day, min_periods=12).std())
    for name, n in [("24h", day), ("7d", 7 * day)]:
        mean = g["mid"].transform(lambda s, n=n: s.rolling(n, min_periods=12).mean())
        df[f"mid_vs_mean_{name}"] = df["mid"] / mean - 1
    df["sell_per_hr_chg_24h"] = df["sell_per_hr"] / g["sell_per_hr"].shift(day) - 1

    median = g["mid"].transform(lambda s: s.rolling(day, min_periods=12).median())
    ratio = df["mid"] / median
    df["outlier"] = (ratio > OUTLIER_RATIO) | (ratio < 1 / OUTLIER_RATIO)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

    df["tradable"] = (
        df["top_bid"].notna()
        & df["top_ask"].notna()
        & ~df["outlier"]
        & (df["turnover_per_hr"] >= MIN_TURNOVER_PER_HR)
    )
    df["fillable"] = df["tradable"] & (df["est_fill_hours"] <= MAX_FILL_HOURS)
    return df.drop(columns=["r1"])


def add_item_categories(df, item_cats):
    return df.merge(item_cats, on="product_id", how="left")


def add_event_features(df, events):
    events = events.copy()
    for col in ["start", "end"]:
        events[col + "_ts"] = events[col].apply(lambda s: pd.Timestamp(s, tz=TIMEZONE).timestamp())

    times = df[["ts"]].drop_duplicates().sort_values("ts").reset_index(drop=True)
    times["t"] = times["ts"].astype(float)

    for cat in events["category"].unique():
        ev = events[events["category"] == cat].sort_values("start_ts")
        nxt = pd.merge_asof(times[["t"]], ev[["start_ts"]], left_on="t",
                            right_on="start_ts", direction="forward")
        prv = pd.merge_asof(times[["t"]], ev[["start_ts", "end_ts"]], left_on="t",
                            right_on="start_ts", direction="backward")
        times[f"ev_{cat}_hours_until"] = (nxt["start_ts"] - times["t"]) / 3600
        times[f"ev_{cat}_active"] = (prv["end_ts"] >= times["t"]).astype(int)

    df = df.merge(times.drop(columns="t"), on="ts", how="left")

    df["item_event_hours_until"] = np.nan
    df["item_event_active"] = 0
    for cat in events["category"].unique():
        mask = df["category"] == cat
        df.loc[mask, "item_event_hours_until"] = df.loc[mask, f"ev_{cat}_hours_until"]
        df.loc[mask, "item_event_active"] = df.loc[mask, f"ev_{cat}_active"]
    return df


def add_perk_features(df, perks):
    perk_cats = perks.groupby("perk")["category"].apply(set).to_dict()

    def active_categories(mayor_perks, minister_perk):
        names = mayor_perks.split("|") if isinstance(mayor_perks, str) else []
        if isinstance(minister_perk, str):
            names.append(minister_perk)
        out = set()
        for name in names:
            out |= perk_cats.get(name, set())
        return frozenset(out)

    mayor_perks = df["mayor_perks"].astype(object).fillna("")
    minister_perk = df["minister_perk"].astype(object).fillna("")
    keys = list(zip(mayor_perks, minister_perk))
    lookup = {key: active_categories(*key) for key in set(keys)}
    cats = [lookup[key] for key in keys]
    df["item_perk_active"] = [int(c in s) for c, s in zip(df["category"], cats)]
    return df


def build_features(db_path=DB_PATH):
    df = clean(load_snapshots(db_path))
    df = add_mayor_features(df, load_mayor(db_path))
    df = add_market_features(df)
    if os.path.exists(ITEM_CATEGORIES_CSV):
        df = add_item_categories(df, pd.read_csv(ITEM_CATEGORIES_CSV))
        if os.path.exists(EVENTS_CSV):
            df = add_event_features(df, pd.read_csv(EVENTS_CSV))
        if os.path.exists(PERKS_CSV):
            df = add_perk_features(df, pd.read_csv(PERKS_CSV))
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.shape)
    print(df.groupby("ts")["tradable"].sum().tail())
    print(df.loc[df["tradable"], "product_id"].nunique(), "items ever tradable")
    print(df[["time", "mayor", "minister", "hours_since_office"]].drop_duplicates("time").tail())