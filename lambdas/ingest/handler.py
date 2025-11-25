# lambdas/ingest/handler.py
import os, json, time, logging
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

#connecting to AWS services via boto3 client
S3 = boto3.client("s3")
SECRETS = boto3.client("secretsmanager")

#url for alpha vantage (our stock market api service)
ALPHA_BASE = "https://www.alphavantage.co/query"

def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# gets the Alpha Vantage key, it prioritizes fetching from Secrets Manager but for local runs we're pulling from 
# an environment variable for local testing
def get_api_key():
    # Use Secrets Manager for AWS deployment; else read env for local dev
    secret_arn = os.environ.get("AV_SECRET_ARN")
    env_key = os.environ.get("ALPHAVANTAGE_KEY")
    if secret_arn:
        try:
            resp = SECRETS.get_secret_value(SecretId=secret_arn)
            secret_str = resp.get("SecretString") or "{}"
            try:
                data = json.loads(secret_str)
            except json.JSONDecodeError:
                # Secret is a plain string, not JSON
                return secret_str

            if isinstance(data, dict):
                # Accept either {"ALPHAVANTAGE_KEY": "..."} or {"key": "..."}
                return data.get("ALPHAVANTAGE_KEY") or data.get("key")
            # Fallback: treat as raw string
            return str(data)
        except ClientError as e:
            log.error(f"Secrets Manager error: {e}")
            raise

    if env_key:
        return env_key

    raise RuntimeError("No API key found. Set ALPHAVANTAGE_KEY or AV_SECRET_ARN.")


def alpha_daily_adjusted(symbol: str, api_key: str, outputsize: str = "compact") -> dict:
    """
    Calls TIME_SERIES_DAILY_ADJUSTED and returns parsed JSON (dict).
    outputsize: 'compact' (~100 most recent trading days) or 'full' (premium).
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
            # Alpha Vantage returns "Note" when throttled, "Error Message" for bad inputs.
            if "Note" in js or "Information" in js:
                raise RuntimeError(
                    f"Alpha Vantage info for {symbol}: {js.get('Note') or js.get('Information')}"
                )
            if "Error Message" in js:
                raise RuntimeError(f"Alpha Vantage error for {symbol}: {js['Error Message']}")
            if "Time Series (Daily)" not in js:
                raise RuntimeError(f"Unexpected response for {symbol}: keys={list(js.keys())[:5]}")
            return js
        except (HTTPError, URLError) as e:
            wait = 5 * (attempt + 1)
            log.warning(f"HTTP error for {symbol}: {e}. Retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after retries for {symbol}")

def extract_latest_daily_row(symbol: str, av_json: dict) -> dict:
    ts = av_json.get("Time Series (Daily)", {})
    if not ts:
        raise RuntimeError(f"No daily time series for {symbol}")
    latest_day = max(ts.keys()) # 'YYYY-MM-DD'
    rec = ts[latest_day]
    return {
        "symbol": symbol,
        "date": latest_day,
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

def put_json(bucket: str, key: str, obj: dict):
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(obj).encode("utf-8"),
        ContentType="application/json",
    )

# handler is trhe lambda entrypoint:
# - it reads tickers from env
# - fetches the latest Alpha Vantage daily data
# - writes one JSON object per symbol/day under raw/<SYMBOL>/dt=<YYYY-MM-DD>/ingest.json
def handler(event, context):
    bucket = os.environ["BUCKET"] #our case - its tickerflow-data-us-east-1
    tickers = [t.strip().upper() for t in os.environ.get("TICKERS", "AAPL,MSFT").split(",") if t.strip()]
    raw_prefix = os.environ.get("RAW_PREFIX", "raw").rstrip("/")
    store_snapshot = os.environ.get("STORE_SNAPSHOT", "0") == "1"
    outputsize = os.environ.get("OUTPUTSIZE", "compact")
    api_key = get_api_key()

    saved, skipped = [], []
    for i, symbol in enumerate(tickers, 1):
        js = alpha_daily_adjusted(symbol, api_key, outputsize=outputsize)
        latest = extract_latest_daily_row(symbol, js)
        dt = latest["date"]  # YYYY-MM-DD

        # Optional: save full snapshot for debugging/auditing
        if store_snapshot:
            snap_key = f"{raw_prefix}/{symbol}/run_ts={datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}/alphavantage_compact.json"
            put_json(bucket, snap_key, js)

        # Idempotent per-day write (small, normalized "raw" row for the latest trading day)
        day_key = f"{raw_prefix}/{symbol}/dt={dt}/ingest.json"
        if s3_key_exists(bucket, day_key):
            log.info(f"{symbol} {dt}: already exists -> skip")
            skipped.append({"symbol": symbol, "date": dt, "key": day_key})
        else:
            put_json(bucket, day_key, latest)
            log.info(f"{symbol} {dt}: wrote {day_key}")
            saved.append({"symbol": symbol, "date": dt, "key": day_key})

        # spacing the calls so we don't face the rate limiter
        if i < len(tickers):
            time.sleep(12)

    result = {"saved": saved, "skipped": skipped, "count_saved": len(saved), "count_skipped": len(skipped)}
    log.info(json.dumps(result))
    return result

# Allow local runs: `python handler.py`
if __name__ == "__main__":
    # For local runs we're just testing the API call + parsing
    os.environ.setdefault("TICKERS", "AAPL")
    os.environ.setdefault("OUTPUTSIZE", "compact")
    key = os.environ.get("ALPHAVANTAGE_KEY")
    if not key:
        raise SystemExit(
            "Need to set ALPHAVANTAGE_KEY for locxal testing, "
            "or run inside Lambda with AV_SECRET_ARN configured."
        )

    print("Local test run. Fetching AAPL…")
    js = alpha_daily_adjusted("AAPL", key)
    print("Raw top-level keys:", list(js.keys()))
    latest = extract_latest_daily_row("AAPL", js)
    print("Latest normalized row:")
    print(json.dumps(latest, indent=2))

