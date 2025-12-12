# forecast_xgboost.py

'''
This script takes in the bucket, features parquet, model run ID, model prefix, and forecast prefix. 
It loads in the models + metrics from S3, loads the features from S3, picks the latest row per symbol, builds
feature matrix X for those rows, and predicts the net-day log return with XGBoost. It then defines the dates and
writes the forecasts to S3 as a parquet.
'''

import argparse
import io
import json
import logging
from datetime import datetime, timezone, timedelta

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


# ---------- Helpers ----------

def load_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"[FORECAST] Loading features from s3://{bucket}/{key}")
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
    log.info(f"[FORECAST] Loading JSON from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)


def load_model_from_s3(bucket: str, key: str):
    log.info(f"[FORECAST] Loading model from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    buf = io.BytesIO(resp["Body"].read())
    model = joblib_load(buf)
    return model


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    log.info(f"[FORECAST] Writing {len(df)} forecast rows to s3://{bucket}/{key}")
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


# ---------- Core forecasting logic ----------

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
    From the full features df, take the latest row per symbol, and build X matrix
    with columns exactly matching feature_names (training-time order).
    """
    if df.empty:
        raise RuntimeError("Features dataframe is empty; cannot build forecast matrix")

    # Sort and take last row per symbol
    df_sorted = df.sort_values(["symbol", "date"])
    latest = df_sorted.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)

    # Numeric features
    missing_base = [c for c in BASE_FEATURES if c not in latest.columns]
    if missing_base:
        raise RuntimeError(f"Missing expected feature columns: {missing_base}")

    X_num = latest[BASE_FEATURES].astype(float)

    # Symbol one-hot
    sym_dummies = pd.get_dummies(latest["symbol"], prefix="sym")

    X = pd.concat([X_num, sym_dummies], axis=1)

    # Align to training feature order
    X = X.reindex(columns=feature_names, fill_value=0.0)

    return latest, X.values


def generate_forecasts(
    bucket: str,
    features_key: str,
    model_run_id: str,
    model_prefix: str = "models/xgboost",
    forecast_prefix: str = "forecasts",
):
    # 1) Derive model & metrics keys
    model_key = f"{model_prefix}/{model_run_id}/model.joblib"
    metrics_key = f"{model_prefix}/{model_run_id}/metrics.json"

    # 2) Load metrics (for feature_names) & model
    metrics = load_json_from_s3(bucket, metrics_key)
    feature_names = metrics.get("feature_names")
    if not feature_names:
        raise RuntimeError("metrics.json does not contain 'feature_names'; cannot align features")

    model = load_model_from_s3(bucket, model_key)

    # 3) Load features df
    df = load_parquet_from_s3(bucket, features_key)

    # 4) Build latest feature matrix
    latest, X = build_latest_feature_matrix(df, feature_names)

    # 5) Predict
    log.info(f"[FORECAST] Predicting next-day returns for {len(latest)} symbols")
    y_pred = model.predict(X)
    pred_direction = np.sign(y_pred).astype(int)

    # Compute target (next-day) date: as_of_date + 1 calendar day
    as_of_dates = latest["date"]
    target_dates = as_of_dates + pd.to_timedelta(1, unit="D")

    # Predict next-day adj_close using log-return assumption
    pred_adj_close = latest["adj_close"].astype(float) * np.exp(y_pred)

    created_ts = datetime.now(timezone.utc).isoformat()

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

    # 6) Choose partition dt for output (by target_date)
    target_dates_unique = forecast_df["target_date"].unique()
    if len(target_dates_unique) != 1:
        log.warning(
            f"[FORECAST] Multiple target_date values found: {target_dates_unique}. "
            f"Using first for partition key."
        )
    target_dt = target_dates_unique[0]

    out_key = f"{forecast_prefix}/dt={target_dt}/forecasts.parquet"
    write_parquet_to_s3(forecast_df, bucket, out_key)

    log.info(f"[FORECAST] Finished. Wrote forecasts to s3://{bucket}/{out_key}")
    return {
        "bucket": bucket,
        "key": out_key,
        "n_rows": int(len(forecast_df)),
        "model_run_id": model_run_id,
        "target_date": target_dt,
    }


# ---------- CLI entrypoint ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="tickerflow-data-us-east-1")
    parser.add_argument("--features-key", default="features/train.parquet")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-prefix", default="models/xgboost")
    parser.add_argument("--forecast-prefix", default="forecasts")
    args = parser.parse_args()

    result = generate_forecasts(
        bucket=args.bucket,
        features_key=args.features_key,
        model_run_id=args.model_run_id,
        model_prefix=args.model_prefix,
        forecast_prefix=args.forecast_prefix,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
