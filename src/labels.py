import numpy as np
import pandas as pd

from features import build_features

TAX = 0.0125
INTERVAL_SECONDS = 900
TICK = 0.1

ENTRY_COLUMNS = {"instabuy": "instabuy_price", "bid": "top_bid"}
EXIT_COLUMNS = {"instasell": "instasell_price", "ask": "top_ask"}


def entry_price(df, mode, tick=TICK):
    price = df[ENTRY_COLUMNS[mode]]
    return price + tick if mode == "bid" else price


def exit_price_adjust(price, mode, tick=TICK):
    return price - tick if mode == "ask" else price


def price_matrix(df, column):
    wide = df.pivot(index="ts", columns="product_id", values=column)
    grid = np.arange(wide.index.min(), wide.index.max() + INTERVAL_SECONDS, INTERVAL_SECONDS)
    return wide.reindex(grid)


def future_median(wide, horizon_steps, tolerance_steps, min_valid):
    window = 2 * tolerance_steps + 1
    centered = wide.rolling(window, center=True, min_periods=min_valid).median()
    return centered.shift(-horizon_steps)


def add_labels(df, horizon_hours=6, tolerance_minutes=30, min_valid=2, tax=TAX, clip=0.5,
               entry_mode="instabuy", exit_mode="instasell", tick=TICK):
    steps_per_hour = 3600 // INTERVAL_SECONDS
    horizon_steps = int(round(horizon_hours * steps_per_hour))
    tolerance_steps = int(tolerance_minutes * 60 // INTERVAL_SECONDS)

    exit_raw = df[EXIT_COLUMNS[exit_mode]].where(~df["outlier"])
    prices = df[["ts", "product_id"]].assign(exit_price=exit_raw)
    future = future_median(price_matrix(prices, "exit_price"),
                           horizon_steps, tolerance_steps, min_valid)

    rows = future.index.get_indexer(df["ts"])
    cols = future.columns.get_indexer(df["product_id"])

    df = df.copy()
    df["future_price"] = exit_price_adjust(future.to_numpy()[rows, cols], exit_mode, tick)
    entry = entry_price(df, entry_mode, tick).where(~df["outlier"])
    df["net_return"] = df["future_price"] * (1 - tax) / entry - 1
    df["has_label"] = df["net_return"].notna()
    df["net_return_clipped"] = df["net_return"].clip(-clip, clip)
    df["profitable"] = (df["net_return"] > 0).astype(float).where(df["has_label"])
    return df


def label_summary(df):
    d = df[df["has_label"] & df["tradable"]]
    if d.empty:
        return pd.Series(dtype=float)
    r = d["net_return"]
    return pd.Series({
        "labeled_rows": len(d),
        "items": d["product_id"].nunique(),
        "snapshots": d["ts"].nunique(),
        "mean_return": r.mean(),
        "median_return": r.median(),
        "std_return": r.std(),
        "pct_profitable": (r > 0).mean(),
        "pct_above_2pct": (r > 0.02).mean(),
        "p05": r.quantile(0.05),
        "p95": r.quantile(0.95),
    })


def break_even_table(df, tax=TAX, entry_mode="instabuy", exit_mode="instasell", tick=TICK):
    d = df[df["tradable"]].copy()
    exit_now = exit_price_adjust(d[EXIT_COLUMNS[exit_mode]], exit_mode, tick)
    d["round_trip_cost"] = 1 - exit_now * (1 - tax) / entry_price(d, entry_mode, tick)
    table = d.groupby("product_id").agg(
        median_cost=("round_trip_cost", "median"),
        median_turnover_per_hr=("turnover_per_hr", "median"),
        snapshots=("ts", "count"),
    )
    return table.sort_values("median_cost")


if __name__ == "__main__":
    df = add_labels(build_features(), horizon_hours=6)
    print(label_summary(df))
    print(break_even_table(df).head(15))
