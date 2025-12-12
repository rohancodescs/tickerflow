# backfill_forecasts_local.py

import argparse
import io
import json
import logging
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from joblib import load as joblib_load

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

S3 = boto3.client("s3")

# ---------- shared with forecast_xgboost ----------

BASE_FEATURES = [
    "adj_close",
    "volume",
    "log_ret_1d",
    "ret_mean_5",
    "ret_std_5",
    "ret_mean_10",
    "ret_std_10",
    "close_mean_5",
    "close_std_5",
    "close_mean_10",
    "close_std_10",
    "day_of_week",
    "month",
]


def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"[BACKFILL] Loading parquet from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")

    buf = io.BytesIO(resp["Body"].read())
    table = pq.read_table(buf)
    df = table.to_pandas()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_json_from_s3(bucket: str, key: str) -> dict:
    log.info(f"[BACKFILL] Loading JSON from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)


def load_model_from_s3(bucket: str, key: str):
    log.info(f"[BACKFILL] Loading model from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    buf = io.BytesIO(resp["Body"].read())
    return joblib_load(buf)


def build_latest_feature_matrix(df: pd.DataFrame, feature_names: list[str]):
    """
    Same semantics as in forecast_xgboost:
    From the full features df (up to some as_of_date), take the latest row per symbol,
    build X with columns matching feature_names.
    """
    if df.empty:
        raise RuntimeError("Features dataframe is empty; cannot build forecast matrix")

    df_sorted = df.sort_values(["symbol", "date"])
    latest = df_sorted.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)

    missing_base = [c for c in BASE_FEATURES if c not in latest.columns]
    if missing_base:
        raise RuntimeError(f"Missing expected feature columns: {missing_base}")

    X_num = latest[BASE_FEATURES].astype(float)
    sym_dummies = pd.get_dummies(latest["symbol"], prefix="sym")

    X = pd.concat([X_num, sym_dummies], axis=1)
    X = X.reindex(columns=feature_names, fill_value=0.0)

    return latest, X.values


def append_parquet_to_s3(bucket: str, key: str, df_new: pd.DataFrame):
    """
    If forecasts/dt=.../forecasts.parquet exists, append; otherwise create.
    """
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(resp["Body"].read())
        table_existing = pq.read_table(buf)
        df_existing = table_existing.to_pandas()
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        log.info(f"[BACKFILL] Appending to existing {key} (old={len(df_existing)}, new={len(df_new)})")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            df_all = df_new
            log.info(f"[BACKFILL] Creating new {key} with {len(df_new)} rows")
        else:
            raise

    table_out = pa.Table.from_pandas(df_all, preserve_index=False)
    out_buf = io.BytesIO()
    pq.write_table(table_out, out_buf)
    out_buf.seek(0)

    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=out_buf.getvalue(),
        ContentType="application/octet-stream",
    )
    log.info(f"[BACKFILL] Wrote {len(df_all)} total rows to s3://{bucket}/{key}")


def next_trading_date(all_dates: list[pd.Timestamp], as_of: pd.Timestamp) -> pd.Timestamp | None:
    """Smallest trading date > as_of, or None if none exist."""
    for d in all_dates:
        if d > as_of:
            return d
    return None


# ---------- main backfill logic ----------


def backfill_range(
    bucket: str,
    features_key: str,
    model_run_id: str,
    model_prefix: str,
    forecast_prefix: str,
    start_as_of: str,
    end_as_of: str,
):
    # 1) Load model + metrics
    model_key = f"{model_prefix}/{model_run_id}/model.joblib"
    metrics_key = f"{model_prefix}/{model_run_id}/metrics.json"

    metrics = load_json_from_s3(bucket, metrics_key)
    feature_names = metrics.get("feature_names")
    if not feature_names:
        raise RuntimeError("metrics.json does not contain 'feature_names'; cannot align features")

    model = load_model_from_s3(bucket, model_key)

    # 2) Load full features df
    df_feat = load_parquet_from_s3(bucket, features_key)
    df_feat["date"] = pd.to_datetime(df_feat["date"]).dt.normalize()

    all_trade_dates = sorted(df_feat["date"].unique())

    start_ts = pd.to_datetime(start_as_of).normalize()
    end_ts = pd.to_datetime(end_as_of).normalize()

    as_of_dates = [d for d in all_trade_dates if start_ts <= d <= end_ts]
    if not as_of_dates:
        raise RuntimeError(
            f"No trading dates in features between {start_as_of} and {end_as_of}. "
            f"Available date range: {all_trade_dates[0]} — {all_trade_dates[-1]}"
        )

    log.info(f"[BACKFILL] Backfilling for as_of_dates: {as_of_dates}")

    created_ts = datetime.now(timezone.utc).isoformat()
    model_s3_uri = f"s3://{bucket}/{model_key}"

    for as_of in as_of_dates:
        log.info(f"[BACKFILL] Processing as_of_date={as_of.date()}")

        # Restrict to history up to as_of_date
        df_hist = df_feat[df_feat["date"] <= as_of].copy()
        latest, X = build_latest_feature_matrix(df_hist, feature_names)

        y_pred = model.predict(X)
        pred_direction = np.sign(y_pred).astype(int)

        # Next trading date
        tgt = next_trading_date(all_trade_dates, as_of)
        if tgt is None:
            log.warning(f"[BACKFILL] No next trading date after {as_of.date()} – skipping")
            continue

        tgt_str = tgt.strftime("%Y-%m-%d")
        as_of_str = as_of.strftime("%Y-%m-%d")

        pred_adj_close = latest["adj_close"].astype(float).values * np.exp(y_pred)

        forecast_df = pd.DataFrame(
            {
                "symbol": latest["symbol"].values,
                "as_of_date": as_of_str,
                "target_date": tgt_str,
                "adj_close": latest["adj_close"].astype(float).values,
                "pred_log_return": y_pred,
                "pred_direction": pred_direction,
                "pred_adj_close": pred_adj_close,
                "model_run_id": model_run_id,
                "model_s3_key": model_s3_uri,
                "created_ts": created_ts,
            }
        )

        out_key = f"{forecast_prefix}/dt={tgt_str}/forecasts.parquet"
        append_parquet_to_s3(bucket, out_key, forecast_df)

    log.info("[BACKFILL] Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="tickerflow-data-us-east-1")
    parser.add_argument("--features-key", default="features/train.parquet")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-prefix", default="models/xgboost")
    parser.add_argument("--forecast-prefix", default="forecasts")
    parser.add_argument("--start-as-of", required=True, help="Calendar start date, e.g. 2025-11-23")
    parser.add_argument("--end-as-of", required=True, help="Calendar end date, e.g. 2025-11-26")
    args = parser.parse_args()

    backfill_range(
        bucket=args.bucket,
        features_key=args.features_key,
        model_run_id=args.model_run_id,
        model_prefix=args.model_prefix,
        forecast_prefix=args.forecast_prefix,
        start_as_of=args.start_as_of,
        end_as_of=args.end_as_of,
    )


if __name__ == "__main__":
    main()
