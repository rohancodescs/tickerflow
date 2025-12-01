import os
import json
import logging
from datetime import datetime, date, timedelta, timezone

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

S3 = boto3.client("s3")

# --- Config via env vars ---
BUCKET = os.environ.get("BUCKET", "tickerflow-data-us-east-1")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw").rstrip("/")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed").rstrip("/")
FORECAST_PREFIX = os.environ.get("FORECAST_PREFIX", "forecasts").rstrip("/")
FEATURES_KEY = os.environ.get("FEATURES_KEY", "features/train.parquet")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8501")

# How many days of lag for raw data we tolerate (to allow for weekends/holidays)
MAX_LAG_DAYS = int(os.environ.get("MAX_LAG_DAYS", "3"))


def extract_dt_from_key(key: str) -> date | None:
    """
    Extract dt=YYYY-MM-DD from an S3 key like:
      raw/AAPL/dt=2025-11-28/ingest.json
      processed/dt=2025-11-28/ohlcv.parquet
      forecasts/dt=2025-11-29/forecasts.parquet
    """
    token = "dt="
    idx = key.find(token)
    if idx == -1:
        return None
    start = idx + len(token)
    # Expect exactly 'YYYY-MM-DD'
    candidate = key[start:start + 10]
    try:
        return datetime.strptime(candidate, "%Y-%m-%d").date()
    except ValueError:
        return None


def get_latest_dt(bucket: str, prefix: str) -> date | None:
    """
    Scan S3 under the given prefix and return the max dt=YYYY-MM-DD seen in keys.
    """
    paginator = S3.get_paginator("list_objects_v2")
    full_prefix = f"{prefix}/"
    dates: set[date] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dt = extract_dt_from_key(key)
            if dt:
                dates.add(dt)

    if not dates:
        return None
    latest = max(dates)
    log.info(f"Latest dt under {bucket}/{prefix}/ is {latest}")
    return latest


def head_object_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except S3.exceptions.NoSuchKey:
        return False
    except Exception as e:
        # If it's any other error, log and treat as not exists
        log.error(f"Error in head_object for {bucket}/{key}: {e}")
        return False


def run_healthcheck() -> dict:
    now = datetime.now(timezone.utc)
    today = now.date()

    # 1) Discover latest dt values
    raw_dt = get_latest_dt(BUCKET, RAW_PREFIX)
    processed_dt = get_latest_dt(BUCKET, PROCESSED_PREFIX)
    forecast_dt = get_latest_dt(BUCKET, FORECAST_PREFIX)

    checks: dict[str, dict] = {}
    overall_ok = True

    # --- Raw freshness ---
    if raw_dt is None:
        checks["raw_freshness"] = {
            "status": "error",
            "details": "No raw dt= partitions found under S3 prefix",
            "bucket": BUCKET,
            "prefix": RAW_PREFIX,
        }
        overall_ok = False
    else:
        lag_days = (today - raw_dt).days
        raw_ok = lag_days <= MAX_LAG_DAYS
        if not raw_ok:
            overall_ok = False
        checks["raw_freshness"] = {
            "status": "ok" if raw_ok else "stale",
            "latest_dt": raw_dt.isoformat(),
            "today": today.isoformat(),
            "lag_days": lag_days,
            "max_allowed_lag_days": MAX_LAG_DAYS,
        }

    # --- Processed alignment ---
    if processed_dt is None:
        checks["processed_alignment"] = {
            "status": "error",
            "details": "No processed dt= partitions found under S3 prefix",
            "bucket": BUCKET,
            "prefix": PROCESSED_PREFIX,
        }
        overall_ok = False
    elif raw_dt is None:
        # already flagged above, but be explicit
        checks["processed_alignment"] = {
            "status": "unknown",
            "details": "No raw dt available to compare with processed",
            "processed_dt": processed_dt.isoformat(),
        }
    else:
        if processed_dt == raw_dt:
            checks["processed_alignment"] = {
                "status": "ok",
                "details": "processed_dt matches raw_dt",
                "raw_dt": raw_dt.isoformat(),
                "processed_dt": processed_dt.isoformat(),
            }
        else:
            overall_ok = False
            checks["processed_alignment"] = {
                "status": "mismatch",
                "details": "processed_dt does not equal raw_dt",
                "raw_dt": raw_dt.isoformat(),
                "processed_dt": processed_dt.isoformat(),
            }

    # --- Forecast alignment ---
    if forecast_dt is None:
        checks["forecast_alignment"] = {
            "status": "error",
            "details": "No forecast dt= partitions found under S3 prefix",
            "bucket": BUCKET,
            "prefix": FORECAST_PREFIX,
        }
        overall_ok = False
    elif processed_dt is None:
        checks["forecast_alignment"] = {
            "status": "unknown",
            "details": "No processed dt available to compare with forecast",
            "forecast_dt": forecast_dt.isoformat(),
        }
    else:
        expected_min_target = processed_dt + timedelta(days=1)
        if forecast_dt >= expected_min_target:
            checks["forecast_alignment"] = {
                "status": "ok",
                "forecast_dt": forecast_dt.isoformat(),
                "expected_min_target_dt": expected_min_target.isoformat(),
            }
        else:
            overall_ok = False
            checks["forecast_alignment"] = {
                "status": "lagging",
                "forecast_dt": forecast_dt.isoformat(),
                "expected_min_target_dt": expected_min_target.isoformat(),
            }

    # --- Features file exists? ---
    features_exists = head_object_exists(BUCKET, FEATURES_KEY)
    if not features_exists:
        overall_ok = False
    checks["features_file"] = {
        "status": "ok" if features_exists else "missing",
        "bucket": BUCKET,
        "key": FEATURES_KEY,
    }

    # High-level status
    status = "ok" if overall_ok else "degraded"

    return {
        "status": status,
        "now_utc": now.isoformat(),
        "latest": {
            "raw_dt": raw_dt.isoformat() if raw_dt else None,
            "processed_dt": processed_dt.isoformat() if processed_dt else None,
            "forecast_dt": forecast_dt.isoformat() if forecast_dt else None,
        },
        "checks": checks,
        "dashboard_url": DASHBOARD_URL,
    }


def lambda_handler(event, context):
    """
    Lambda entrypoint.
    """
    result = run_healthcheck()
    # For Step Functions or console, returning just the dict is fine.
    # If you later front this with API Gateway, you can wrap in statusCode/body.
    return result
