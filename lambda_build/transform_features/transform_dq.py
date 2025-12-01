# transform_dq.py
#
# Pipeline:
#   1) Read raw ingest.json files from S3 under raw/*/dt=*/ingest.json
#   2) Run basic data-quality checks
#   3) Write cleaned OHLCV per day to processed/dt=YYYY-MM-DD/ohlcv.parquet
#   4) Load ALL processed parquet, build ML features + targets
#   5) Write features/train.parquet to S3

import os
import io
import json
import logging

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

S3 = boto3.client("s3")

BUCKET = os.environ.get("BUCKET", "tickerflow-data-us-east-1")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw").rstrip("/")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed").rstrip("/")
FEATURES_KEY = os.environ.get("FEATURES_KEY", "features/train.parquet")


# ---------------- RAW → DATAFRAME ----------------

def list_raw_ingest_keys(bucket: str, raw_prefix: str):
    """
    Iterate over all raw ingest.json objects under raw/*/dt=*/ingest.json
    """
    paginator = S3.get_paginator("list_objects_v2")
    prefix = f"{raw_prefix}/"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/ingest.json"):
                yield key


def load_raw_rows(bucket: str, raw_prefix: str) -> pd.DataFrame:
    """
    Load all ingest.json rows from S3 into a DataFrame.
    """
    rows = []
    for key in list_raw_ingest_keys(bucket, raw_prefix):
        try:
            resp = S3.get_object(Bucket=bucket, Key=key)
            body = resp["Body"].read()
            rec = json.loads(body)
            rows.append(rec)
        except ClientError as e:
            log.error(f"Error reading {key}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ensure date is datetime; raw is 'YYYY-MM-DD'
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    return df


# ---------------- DATA QUALITY ----------------

def run_data_quality_checks(df: pd.DataFrame):
    """
    Run basic DQ checks. Returns (ok: bool, report: dict)
    """
    report = {"errors": [], "warnings": [], "row_count": int(len(df))}
    required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]

    if df.empty:
        report["errors"].append("DataFrame is empty (no raw rows found).")
        return False, report

    # 1) Columns present
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        report["errors"].append(f"Missing required columns: {missing}")
        return False, report

    # 2) Date parsing
    if df["date"].isna().any():
        n = int(df["date"].isna().sum())
        report["errors"].append(f"'date' column has {n} unparsable values.")

    # 3) Null checks
    for col in required_cols:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            report["errors"].append(f"Column '{col}' has {null_count} null values.")

    # 4) Value constraints
    for col in ["open", "high", "low", "close"]:
        bad = int((df[col] <= 0).sum())
        if bad > 0:
            report["errors"].append(f"Column '{col}' has {bad} non-positive values.")

    bad_high_low = int((df["high"] < df["low"]).sum())
    if bad_high_low > 0:
        report["errors"].append(f"{bad_high_low} rows where high < low.")

    bad_volume = int((df["volume"] < 0).sum())
    if bad_volume > 0:
        report["errors"].append(f"{bad_volume} rows with negative volume.")

    # 5) Uniqueness of (symbol, date)
    dupes = int(df.duplicated(subset=["symbol", "date"]).sum())
    if dupes > 0:
        report["warnings"].append(f"{dupes} duplicate (symbol, date) rows.")

    ok = len(report["errors"]) == 0
    return ok, report


# ---------------- WRITE PROCESSED PARQUET ----------------

def write_processed_parquet(df: pd.DataFrame, bucket: str, processed_prefix: str):
    """
    Write processed data to S3 as partitioned Parquet:
      processed/dt=YYYY-MM-DD/ohlcv.parquet
    """
    if df.empty:
        log.warning("No rows to write to processed/")
        return 0

    # Ensure required columns exist
    cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividend",
        "split_coef",
        "source",
        "ingest_ts",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"])

    written = 0
    # Group by date and write one partition per day
    for dt, group in df.groupby(df["date"].dt.date):
        out_key = f"{processed_prefix}/dt={dt}/ohlcv.parquet"
        table = pa.Table.from_pandas(group, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)

        S3.put_object(
            Bucket=bucket,
            Key=out_key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        log.info(f"Wrote {len(group)} rows to s3://{bucket}/{out_key}")
        written += len(group)

    return written


# ---------------- LOAD PROCESSED & BUILD FEATURES ----------------

def list_processed_parquet_keys(bucket: str, processed_prefix: str):
    keys = []
    paginator = S3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{processed_prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                keys.append(key)
    return keys


def load_all_processed(bucket: str, processed_prefix: str) -> pd.DataFrame:
    keys = list_processed_parquet_keys(bucket, processed_prefix)
    if not keys:
        log.warning("No processed parquet found.")
        return pd.DataFrame()

    frames = []
    for key in keys:
        resp = S3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        table = pq.read_table(buf)
        frames.append(table.to_pandas())

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature set similar to what you've been using:
      - log_ret_1d
      - rolling means/stds of returns (5, 10, 20)
      - rolling means/stds of close (5, 10)
      - calendar features
      - next-day target_log_return + target_direction
    """
    if df.empty:
        log.warning("No processed data to build features from.")
        return df

    df = df.sort_values(["symbol", "date"]).copy()

    # 1-day log return
    df["log_ret_1d"] = df.groupby("symbol")["adj_close"].apply(
        lambda s: np.log(s / s.shift(1))
    )

    def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
        r = g["log_ret_1d"]

        for w in (5, 10, 20):
            g[f"ret_mean_{w}"] = r.rolling(w).mean()
            g[f"ret_std_{w}"] = r.rolling(w).std()

        close = g["close"]
        for w in (5, 10):
            rc = close.rolling(w)
            g[f"close_mean_{w}"] = rc.mean()
            g[f"close_std_{w}"] = rc.std()

        return g

    df = df.groupby("symbol", group_keys=False).apply(add_rolling_features)

    # Calendar features
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month

    # Next-day target
    df["target_log_return"] = df.groupby("symbol")["log_ret_1d"].shift(-1)
    df["target_direction"] = np.sign(df["target_log_return"])

    return df


def write_features_parquet(df_feat: pd.DataFrame, bucket: str, key: str):
    if df_feat.empty:
        log.warning("Feature DataFrame is empty; not writing train.parquet.")
        return

    table = pa.Table.from_pandas(df_feat, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    log.info(f"Wrote {len(df_feat)} feature rows to s3://{bucket}/{key}")


# ---------------- MAIN ----------------

def main():
    log.info(f"Loading raw rows from s3://{BUCKET}/{RAW_PREFIX}/")
    df_raw = load_raw_rows(BUCKET, RAW_PREFIX)
    log.info(f"Loaded {len(df_raw)} raw rows")

    if df_raw.empty:
        log.error("No raw rows found. Exiting.")
        return

    ok, report = run_data_quality_checks(df_raw)

    log.info("DQ report:")
    log.info(json.dumps(report, indent=2, default=str))

    if not ok:
        log.error("Data quality check FAILED. Not writing processed or features.")
        return

    log.info("Data quality passed. Writing processed Parquet.")
    written = write_processed_parquet(df_raw, BUCKET, PROCESSED_PREFIX)
    log.info(f"Transform complete, wrote {written} processed rows.")

    # Build features from all processed data
    log.info("Loading processed Parquet to build features…")
    df_proc = load_all_processed(BUCKET, PROCESSED_PREFIX)
    log.info(f"Loaded {len(df_proc)} processed rows")

    df_feat = build_features(df_proc)
    log.info(f"Built features with {len(df_feat)} rows")

    write_features_parquet(df_feat, BUCKET, FEATURES_KEY)
    log.info("Transform + DQ + feature build complete.")


if __name__ == "__main__":
    main()
