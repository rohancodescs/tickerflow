import os
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

# Defaults (can be overridden by environment variables or event)
BUCKET = os.environ.get("BUCKET", "tickerflow-data-us-east-1")
FEATURES_KEY = os.environ.get("FEATURES_KEY", "features/train.parquet")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/xgboost")
FORECAST_PREFIX = os.environ.get("FORECAST_PREFIX", "forecasts")

# ---------- S3 helpers ----------

def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info("[FORECAST] Loading features from s3://%s/%s", bucket, key)
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")

    buf = io.BytesIO(resp["Body"].read())
    table = pq.read_table(buf)
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_json_from_s3(bucket: str, key: str) -> dict:
    log.info("[FORECAST] Loading JSON from s3://%s/%s", bucket, key)
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)


def load_model_from_s3(bucket: str, key: str):
    log.info("[FORECAST] Loading model from s3://%s/%s", bucket, key)
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    buf = io.BytesIO(resp["Body"].read())
    model = joblib_load(buf)
    return model


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    log.info("[FORECAST] Writing %d forecast rows to s3://%s/%s", len(df), bucket, key)
    table = pa.Table.from_pandas(df)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)
    S3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )


# ---------- Feature matrix construction ----------

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


def build_latest_feature_matrix(df: pd.DataFrame, feature_names: list[str]):
    """
    Take the latest row per symbol and build X with columns matching training-time feature_names.
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

    # Align to training feature order
    X = X.reindex(columns=feature_names, fill_value=0.0)

    return latest, X.values


def find_latest_model_run_id(bucket: str, model_prefix: str) -> str:
    """
    Inspect s3://bucket/model_prefix/* and return the lexicographically latest run_id.
    Assumes run_id folders look like: models/xgboost/20251128T052722Z/...
    """
    prefix = model_prefix.rstrip("/") + "/"
    paginator = S3.get_paginator("list_objects_v2")
    run_ids: set[str] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            suffix = key[len(prefix) :]  # e.g. "20251128T052722Z/model.joblib"
            parts = suffix.split("/")
            if parts and parts[0]:
                run_ids.add(parts[0])

    if not run_ids:
        raise RuntimeError(f"No model runs found under s3://{bucket}/{prefix}")

    latest = sorted(run_ids)[-1]
    log.info("[FORECAST] Auto-selected latest model_run_id=%s", latest)
    return latest


# ---------- Core forecasting logic ----------

def generate_forecasts(
    bucket: str,
    features_key: str,
    model_run_id: str,
    model_prefix: str = MODEL_PREFIX,
    forecast_prefix: str = FORECAST_PREFIX,
):
    model_prefix = model_prefix.rstrip("/")
    forecast_prefix = forecast_prefix.rstrip("/")

    model_key = f"{model_prefix}/{model_run_id}/model.joblib"
    metrics_key = f"{model_prefix}/{model_run_id}/metrics.json"

    # Get feature order + model
    metrics = load_json_from_s3(bucket, metrics_key)
    feature_names = metrics.get("feature_names")
    if not feature_names:
        raise RuntimeError("metrics.json does not contain 'feature_names'; cannot align features")

    model = load_model_from_s3(bucket, model_key)

    # Load features and build X
    df = load_parquet_from_s3(bucket, features_key)
    latest, X = build_latest_feature_matrix(df, feature_names)

    log.info("[FORECAST] Predicting next-day returns for %d symbols", len(latest))
    y_pred = model.predict(X)
    pred_direction = np.sign(y_pred).astype(int)

    as_of_dates = latest["date"]
    target_dates = as_of_dates + pd.to_timedelta(1, unit="D")
    pred_adj_close = latest["adj_close"].astype(float) * np.exp(y_pred)

    created_ts = datetime.now(timezone.utc).isoformat()

    # IMPORTANT: these column names must match your Athena `tickerflow_forecasts` table
    forecast_df = pd.DataFrame(
        {
            "symbol": latest["symbol"],
            "as_of_date": as_of_dates.dt.strftime("%Y-%m-%d"),
            "target_date": target_dates.dt.strftime("%Y-%m-%d"),
            "adj_close": latest["adj_close"].astype(float),
            "pred_log_return": y_pred,
            "pred_direction": pred_direction,
            "pred_adj_close": pred_adj_close,
            "model_run_id": model_run_id,
            "model_s3_key": f"s3://{bucket}/{model_key}",
            "created_ts": created_ts,
        }
    )

    # One partition per target_date (normally just one)
    target_dates_unique = forecast_df["target_date"].unique()
    if len(target_dates_unique) != 1:
        log.warning(
            "[FORECAST] Multiple target_date values found: %s. Using first for partition key.",
            target_dates_unique,
        )
    target_dt = target_dates_unique[0]

    out_key = f"{forecast_prefix}/dt={target_dt}/forecasts.parquet"
    write_parquet_to_s3(forecast_df, bucket, out_key)

    log.info("[FORECAST] Finished. Wrote forecasts to s3://%s/%s", bucket, out_key)
    return {
        "bucket": bucket,
        "key": out_key,
        "n_rows": int(len(forecast_df)),
        "model_run_id": model_run_id,
        "target_date": target_dt,
    }


# ---------- Lambda entrypoint ----------

def handler(event, context):
    """
    Lambda entrypoint.

    event can optionally include:
      - bucket
      - features_key
      - model_prefix
      - forecast_prefix
      - model_run_id

    If model_run_id is not provided (event or env), we'll auto-pick the latest.
    """
    if not isinstance(event, dict):
        event = {}

    bucket = event.get("bucket", BUCKET)
    features_key = event.get("features_key", FEATURES_KEY)
    model_prefix = event.get("model_prefix", MODEL_PREFIX)
    forecast_prefix = event.get("forecast_prefix", FORECAST_PREFIX)

    model_run_id = event.get("model_run_id") or os.environ.get("MODEL_RUN_ID")
    if not model_run_id:
        model_run_id = find_latest_model_run_id(bucket, model_prefix)

    result = generate_forecasts(
        bucket=bucket,
        features_key=features_key,
        model_run_id=model_run_id,
        model_prefix=model_prefix,
        forecast_prefix=forecast_prefix,
    )
    log.info("[FORECAST] Lambda completed: %s", result)
    return result
