# transform_dq.py
import os
import io
import json
import logging
from datetime import datetime

import boto3
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


def load_raw_rows(bucket: str, raw_prefix: str):
    """
    Load all ingest.json rows from S3 into a list of dicts.
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
    return rows


def run_data_quality_checks(df: pd.DataFrame):
    """
    Run basic DQ checks. Returns (ok: bool, report: dict)
    """
    report = {"errors": [], "warnings": [], "row_count": len(df)}
    required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]

    # 1) Columns present
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        report["errors"].append(f"Missing required columns: {missing}")

    if df.empty:
        report["errors"].append("DataFrame is empty (no raw rows found).")
        return False, report

    # 2) Convert date to datetime
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except Exception as e:
        report["errors"].append(f"Failed to parse 'date' column as YYYY-MM-DD: {e}")

    # 3) Null checks
    for col in required_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            report["errors"].append(f"Column '{col}' has {null_count} null values.")

    # 4) Value constraints
    for col in ["open", "high", "low", "close"]:
        bad = (df[col] <= 0).sum()
        if bad > 0:
            report["errors"].append(f"Column '{col}' has {bad} non-positive values.")

    bad_high_low = (df["high"] < df["low"]).sum()
    if bad_high_low > 0:
        report["errors"].append(f"{bad_high_low} rows where high < low.")

    bad_volume = (df["volume"] < 0).sum()
    if bad_volume > 0:
        report["errors"].append(f"{bad_volume} rows with negative volume.")

    # 5) Uniqueness of (symbol, date)
    dupes = df.duplicated(subset=["symbol", "date"]).sum()
    if dupes > 0:
        report["errors"].append(f"{dupes} duplicate (symbol, date) rows.")

    ok = len(report["errors"]) == 0
    return ok, report


def write_processed_parquet(df: pd.DataFrame, bucket: str, processed_prefix: str):
    """
    Write processed data to S3 as partitioned Parquet:
      processed/dt=YYYY-MM-DD/ohlcv.parquet
    """
    # Ensure column order
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

    df = df[cols]

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


def main():
    log.info(f"Loading raw rows from s3://{BUCKET}/{RAW_PREFIX}/")
    rows = load_raw_rows(BUCKET, RAW_PREFIX)
    log.info(f"Loaded {len(rows)} raw rows")

    if not rows:
        log.error("No raw rows found. Exiting.")
        return

    df = pd.DataFrame(rows)
    ok, report = run_data_quality_checks(df)

    log.info("DQ report:")
    log.info(json.dumps(report, indent=2, default=str))

    if not ok:
        log.error("Data quality check FAILED. Not writing processed data.")
        return

    log.info("Data quality passed. Writing processed Parquet.")
    write_processed_parquet(df, BUCKET, PROCESSED_PREFIX)
    log.info("Transform + DQ complete.")


if __name__ == "__main__":
    main()
