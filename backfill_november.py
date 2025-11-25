# backfill_november.py
# THIS FILE SERVES TO BE RAN ONCE; handler.py contains the logic for pulling market data daily from 11/24 onwards
# Since we wil be training a model for stock prediction, this script was ran to fill data in the S3 bucket from 11/1-11/23
import os
import json
import time
import logging
from datetime import datetime, date, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

S3 = boto3.client("s3")

BUCKET = "tickerflow-data-us-east-1"
RAW_PREFIX = "raw"
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT"]

START_DATE = date(2025, 11, 1)
END_DATE = date(2025, 11, 23)

ALPHA_BASE = "https://www.alphavantage.co/query"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def alpha_daily_adjusted(symbol: str, api_key: str, outputsize: str = "full") -> dict:
    """
    Fetch full daily time series for a symbol.
    outputsize:
      - 'full' for entire history (requires paid key)
      - 'compact' for ~100 latest days (works on free)
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "json",
        "apikey": api_key,
    }
    url = f"{ALPHA_BASE}?{urlencode(params)}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=20) as r:
                body = r.read().decode("utf-8")
            js = json.loads(body)
            if "Note" in js:
                wait = 15 * (attempt + 1)
                log.warning(f"[{symbol}] Rate limit note from Alpha Vantage. Sleeping {wait}s then retrying.")
                time.sleep(wait)
                continue
            if "Error Message" in js:
                raise RuntimeError(f"Alpha Vantage error for {symbol}: {js['Error Message']}")
            if "Time Series (Daily)" not in js:
                raise RuntimeError(f"Unexpected response for {symbol}: keys={list(js.keys())[:5]}")
            return js
        except (HTTPError, URLError) as e:
            wait = 5 * (attempt + 1)
            log.warning(f"[{symbol}] HTTP error: {e}. Retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after retries for {symbol}")


def normalize_row(symbol: str, date_str: str, rec: dict) -> dict:
    """Make a row that matches your Lambda's ingest.json schema."""
    return {
        "symbol": symbol,
        "date": date_str,
        "open": float(rec["1. open"]),
        "high": float(rec["2. high"]),
        "low": float(rec["3. low"]),
        "close": float(rec["4. close"]),
        "adj_close": float(rec.get("5. adjusted close", rec["4. close"])),
        "volume": int(rec["6. volume"]),
        "dividend": float(rec.get("7. dividend amount", 0.0)),
        "split_coef": float(rec.get("8. split coefficient", 1.0)),
        "source": "alpha_vantage",
        "ingest_ts": _now_iso(),
    }


def s3_key_exists(bucket: str, key: str) -> bool:
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def put_json(bucket: str, key: str, obj: dict) -> None:
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj).encode("utf-8"),
        ContentType="application/json",
    )


def backfill_symbol(symbol: str, api_key: str, start: date, end: date, outputsize: str = "full") -> None:
    js = alpha_daily_adjusted(symbol, api_key, outputsize=outputsize)
    ts = js.get("Time Series (Daily)", {})
    if not ts:
        log.warning(f"[{symbol}] No time series returned.")
        return

    count_saved = 0
    count_skipped = 0

    for d_str, rec in ts.items():
        d = datetime.strptime(d_str, "%Y-%m-%d").date()
        if d < start or d > end:
            continue  # outside November (or given range)

        day_key = f"{RAW_PREFIX}/{symbol}/dt={d_str}/ingest.json"
        if s3_key_exists(BUCKET, day_key):
            log.info(f"[{symbol}] {d_str}: already exists -> skip")
            count_skipped += 1
            continue

        row = normalize_row(symbol, d_str, rec)
        put_json(BUCKET, day_key, row)
        log.info(f"[{symbol}] {d_str}: wrote {day_key}")
        count_saved += 1

    log.info(f"[{symbol}] Backfill complete. Saved={count_saved}, Skipped={count_skipped}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

    api_key = os.environ.get("ALPHAVANTAGE_KEY")
    if not api_key:
        raise SystemExit("Set ALPHAVANTAGE_KEY in your shell before running this script.")

    outputsize = os.environ.get("OUTPUTSIZE", "full")

    log.info(f"Backfilling for dates {START_DATE} to {END_DATE} into bucket={BUCKET}, prefix={RAW_PREFIX}")
    for symbol in TICKERS:
        log.info(f"Starting backfill for {symbol}")
        backfill_symbol(symbol, api_key, start=START_DATE, end=END_DATE, outputsize=outputsize)
        time.sleep(1)

    log.info("Backfill finished.")
