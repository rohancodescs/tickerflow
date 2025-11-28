import os
import io
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from joblib import load

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

S3 = boto3.client("s3")

# ==== Config via env vars ====
BUCKET = os.environ.get("BUCKET", "tickerflow-data-us-east-1")

# Where features live (same as training)
FEATURES_KEY = os.environ.get("FEATURES_KEY", "features/train.parquet")

# Which trained model run to use
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models")
MODEL_RUN_ID = os.environ["MODEL_RUN_ID"]  # e.g. 20251128T000843Z from your training run

MODEL_BASE = f"{MODEL_PREFIX}/xgboost/{MODEL_RUN_ID}"
MODEL_KEY = os.environ.get("MODEL_KEY", f"{MODEL_BASE}/model.joblib")
METRICS_KEY = os.environ.get("METRICS_KEY", f"{MODEL_BASE}/metrics.json")

# Where to write forecasts
FORECAST_PREFIX = os.environ.get("FORECAST_PREFIX", "curated/forecasts").rstrip("/")


def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"Loading parquet from s3://{bucket}/{key}")
    resp = S3.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    table = pq.read_table(buf)
    return table.to_pandas()


def load_model_from_s3(bucket: str, key: str):
    log.info(f"Loading model from s3://{bucket}/{key}")
    resp = S3.get_object(Bucket=bucket, Key=key)
    buf = io.BytesIO(resp["Body"].read())
    buf.seek(0)
    return load(buf)


def load_metrics_from_s3(bucket: str, key: str) -> dict:
    log.info(f"Loading metrics from s3://{bucket}/{key}")
    resp = S3.get_object(Bucket=bucket, Key=key)
    data = resp["Body"].read()
    return json.loads(data)


def build_feature_matrix_for_date(df: pd.DataFrame, target_date: pd.Timestamp, feature_names=None):
    """
    Rebuild the feature matrix for a single as-of date using the same
    feature engineering approach as training.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Filter to the target date (as-of date)
    mask = df["date"].dt.normalize() == target_date.normalize()
    df_day = df.loc[mask].copy()
    if df_day.empty:
        raise RuntimeError(f"No rows found for as-of date {target_date.date()} in features dataset")

    # Same feature list used in training (minus 20-day windows)
    base_features = [
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
    used = [c for c in base_features if c in df_day.columns]

    X_num = df_day[used].astype(float)
    sym_dummies = pd.get_dummies(df_day["symbol"], prefix="sym")
    X = pd.concat([X_num, sym_dummies], axis=1)

    # If we have feature_names from training, align columns to that order
    if feature_names is not None:
        for c in feature_names:
            if c not in X.columns:
                X[c] = 0.0
        # drop any extra cols not seen during training
        X = X[feature_names]

    meta = df_day[["symbol", "date", "adj_close"]].copy()
    return X.values, meta


def write_forecasts_to_s3(df_forecast: pd.DataFrame, bucket: str, prefix: str, as_of_date: pd.Timestamp):
    dt_str = as_of_date.strftime("%Y-%m-%d")
    out_key = f"{prefix}/dt={dt_str}/forecasts.parquet"

    table = pa.Table.from_pandas(df_forecast, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    S3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    log.info(f"Wrote {len(df_forecast)} forecasts to s3://{bucket}/{out_key}")
    return out_key


def write_forecast_run_metadata(bucket: str, prefix: str, as_of_date: pd.Timestamp, run_info: dict):
    dt_str = as_of_date.strftime("%Y-%m-%d")
    out_key = f"{prefix}/dt={dt_str}/forecast_run.json"
    S3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=json.dumps(run_info, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"Wrote forecast run metadata to s3://{bucket}/{out_key}")
    return out_key


def main():
    t_total_start = time.perf_counter()

    # 1) Load metrics (to get feature_names, etc.)
    metrics = load_metrics_from_s3(BUCKET, METRICS_KEY)
    feature_names = metrics.get("feature_names", None)

    # 2) Load model
    model = load_model_from_s3(BUCKET, MODEL_KEY)

    # 3) Load full features dataset
    df_feats = load_parquet_from_s3(BUCKET, FEATURES_KEY)
    df_feats["date"] = pd.to_datetime(df_feats["date"])

    latest_date = df_feats["date"].max().normalize()
    forecast_for_date = latest_date + timedelta(days=1)

    log.info(f"As-of date = {latest_date.date()}, forecasting for {forecast_for_date.date()}")

    # 4) Build feature matrix for latest date
    t_feat_start = time.perf_counter()
    X_latest, meta_latest = build_feature_matrix_for_date(df_feats, latest_date, feature_names=feature_names)
    t_feat = time.perf_counter() - t_feat_start

    # 5) Run model inference
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_latest)
    t_pred = time.perf_counter() - t_pred_start

    # 6) Build forecast dataframe
    pred_direction = np.sign(y_pred).astype(float)
    df_forecast = pd.DataFrame(
        {
            "symbol": meta_latest["symbol"].values,
            "as_of_date": latest_date.date().isoformat(),
            "forecast_for_date": forecast_for_date.date().isoformat(),
            "latest_adj_close": meta_latest["adj_close"].astype(float).values,
            "predicted_log_return": y_pred,
            "predicted_direction": pred_direction,
            "model_run_id": MODEL_RUN_ID,
        }
    )

    # 7) Write forecasts & run metadata
    forecasts_key = write_forecasts_to_s3(df_forecast, BUCKET, FORECAST_PREFIX, latest_date)

    t_total = time.perf_counter() - t_total_start
    run_info = {
        "model_run_id": MODEL_RUN_ID,
        "as_of_date": latest_date.date().isoformat(),
        "forecast_for_date": forecast_for_date.date().isoformat(),
        "forecasts_key": f"s3://{BUCKET}/{forecasts_key}",
        "timing_seconds": {
            "feature_prep": t_feat,
            "prediction": t_pred,
            "total": t_total,
        },
        "n_symbols": int(len(df_forecast)),
        "status": "success",
    }

    write_forecast_run_metadata(BUCKET, FORECAST_PREFIX, latest_date, run_info)


if __name__ == "__main__":
    main()
