"""Aggregate raw ShortVideo interactions into per-video behavior labels.

Input:
  interaction.csv with columns (tiny dataset schema):
    user_id, pid, exposed_time, p_date, p_hour, watch_time, duration,
    cvm_like, click, comment, follow, collect, forward, hate

  If the column 'effective_view' is absent (tiny dataset), it is derived
  from 'click': click==True means the user actively watched, which is the
  closest proxy available.

Output:
  data_processed/behavior/video_behavior_labels.parquet
  one row per pid with aggregated rates and stats.

Usage:
  python scripts/build_behavior_labels.py \
      --interaction data_raw/shortvideo_tiny/interaction.csv \
      --out data_processed/behavior/video_behavior_labels.parquet \
      --min-impressions 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BOOL_BEHAVIORS = [
    "cvm_like",
    "comment",
    "follow",
    "collect",
    "forward",
    "hate",
    "effective_view",
]

RATE_RENAME = {
    "cvm_like": "like_rate",
    "comment": "comment_rate",
    "follow": "follow_rate",
    "collect": "collect_rate",
    "forward": "forward_rate",
    "hate": "hate_rate",
    "effective_view": "effective_view_rate",
}


def coerce_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce a column to 0/1 floats regardless of input dtype."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.float32)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(np.float32) > 0).astype(np.float32)
    return series.astype(str).str.lower().isin({"true", "1", "t", "yes"}).astype(np.float32)


def aggregate(df: pd.DataFrame, min_impressions: int) -> pd.DataFrame:
    # tiny dataset lacks 'effective_view'; per README it means watch_time >= 3s
    if "effective_view" not in df.columns:
        if "watch_time" in df.columns:
            df["effective_view"] = (pd.to_numeric(df["watch_time"], errors="coerce").fillna(0) >= 3).astype(np.float32)
            print("derived 'effective_view' as watch_time >= 3s")
        elif "click" in df.columns:
            df["effective_view"] = df["click"]
            print("derived 'effective_view' from 'click' column (fallback)")
        else:
            raise KeyError("need 'effective_view', 'watch_time', or 'click' column")

    for col in BOOL_BEHAVIORS:
        if col not in df.columns:
            raise KeyError(f"missing required column: {col}")
        df[col] = coerce_bool(df[col])

    if "watch_time" not in df.columns:
        raise KeyError("missing required column: watch_time")
    df["watch_time"] = pd.to_numeric(df["watch_time"], errors="coerce").fillna(0.0)

    agg_dict = {col: "mean" for col in BOOL_BEHAVIORS}
    agg_dict["watch_time"] = ["mean", "median", "std"]
    agg_dict["user_id"] = "count"

    grouped = df.groupby("pid").agg(agg_dict)
    grouped.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns
    ]
    grouped = grouped.rename(columns={"user_id_count": "n_impressions"})
    grouped = grouped.rename(columns={
        "watch_time_mean": "mean_watch_time",
        "watch_time_median": "median_watch_time",
        "watch_time_std": "std_watch_time",
    })
    # after flattening, single-agg cols get a _mean suffix; map accordingly
    rate_rename_flat = {f"{k}_mean": v for k, v in RATE_RENAME.items()}
    rate_rename_flat.update(RATE_RENAME)  # also handle no-suffix variant
    grouped = grouped.rename(columns=rate_rename_flat)

    grouped = grouped.reset_index()
    before = len(grouped)
    grouped = grouped[grouped["n_impressions"] >= min_impressions].copy()
    after = len(grouped)
    print(f"filtered {before - after} / {before} videos with < {min_impressions} impressions")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction", type=Path, required=True,
                        help="Path to interaction_filtered.csv")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output parquet path")
    parser.add_argument("--min-impressions", type=int, default=10,
                        help="Drop videos with fewer than this many impressions")
    parser.add_argument("--chunksize", type=int, default=2_000_000,
                        help="Read CSV in chunks; lower if memory tight")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.interaction.exists():
        raise FileNotFoundError(args.interaction)

    # streaming read to keep peak memory down on large files
    print(f"reading {args.interaction}")
    parts = []
    for i, chunk in enumerate(pd.read_csv(args.interaction, chunksize=args.chunksize)):
        print(f"  chunk {i}: {len(chunk):,} rows")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    print(f"total rows: {len(df):,}")
    print(f"unique pids: {df['pid'].nunique():,}")
    print(f"unique users: {df['user_id'].nunique():,}")

    labels = aggregate(df, args.min_impressions)
    print(f"final label rows: {len(labels):,}")
    print("\nlabel summary:")
    print(labels.describe(percentiles=[0.5, 0.9, 0.99]).round(4))

    labels.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
