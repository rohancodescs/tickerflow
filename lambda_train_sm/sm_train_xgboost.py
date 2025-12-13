# sm_train_xgboost.py
import argparse
import io
import json
import logging
import os
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from joblib import dump
from xgboost import XGBRegressor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

S3 = boto3.client("s3")


# ---------- helpers ----------

def load_features_from_s3(bucket: str, key: str) -> pd.DataFrame:
    log.info(f"[TRAIN] Loading features from s3://{bucket}/{key}")
    try:
        resp = S3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"Failed to get {key} from {bucket}: {e}")
    buf = io.BytesIO(resp["Body"].read())
    table = pq.read_table(buf)
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


# this function ensures we have a target_log_return column, if its missing we derive it from adj_close as next-day log returns
# and then builds X, y, meta, and feature_names
def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"])

    if "target_log_return" not in df.columns:
        log.info(
            "[TRAIN] 'target_log_return' not found; computing from adj_close "
            "as next-day log return per symbol."
        )
        if "adj_close" not in df.columns:
            raise RuntimeError(
                "Cannot compute target_log_return: 'adj_close' column is missing."
            )

        # target_log_return_t = log(adj_close_{t+1} / adj_close_t)
        df["target_log_return"] = df.groupby("symbol")["adj_close"].transform(
            lambda s: np.log(s.shift(-1) / s)
        )

    # Drop rows without a target (e.g., last date per symbol)
    df = df.dropna(subset=["target_log_return"]).copy()

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
    feature_cols = [c for c in base_features if c in df.columns]

    X_num = df[feature_cols].astype(float)
    sym_dummies = pd.get_dummies(df["symbol"], prefix="sym")
    X = pd.concat([X_num, sym_dummies], axis=1)

    y = df["target_log_return"].astype(float).values
    meta = df[["date", "symbol"]].copy()

    log.info(
        f"[TRAIN] Using {len(feature_cols)} numeric features + "
        f"{sym_dummies.shape[1]} symbol dummies"
    )
    return X, y, meta, list(X.columns)


def time_based_split(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame,
                     train_frac: float, val_frac: float):
    dates = np.sort(meta["date"].dt.normalize().unique())
    n_dates = len(dates)
    if n_dates < 5:
        raise RuntimeError(f"Not enough unique dates ({n_dates}) for train/val/test split")

    train_end = int(n_dates * train_frac)
    val_end = int(n_dates * (train_frac + val_frac))
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    def mask_for_dates(date_set):
        return meta["date"].dt.normalize().isin(date_set)

    m_train = mask_for_dates(train_dates)
    m_val = mask_for_dates(val_dates)
    m_test = mask_for_dates(test_dates)

    def subset(mask):
        return X[mask].values, y[mask], meta.loc[mask].copy()

    X_tr, y_tr, meta_tr = subset(m_train)
    X_val, y_val, meta_val = subset(m_val)
    X_te, y_te, meta_te = subset(m_test)

    split_info = {
        "n_dates_total": int(n_dates),
        "n_train_dates": int(len(train_dates)),
        "n_val_dates": int(len(val_dates)),
        "n_test_dates": int(len(test_dates)),
        "n_train_rows": int(len(y_tr)),
        "n_val_rows": int(len(y_val)),
        "n_test_rows": int(len(y_te)),
    }

    log.info(f"[TRAIN] Split info: {split_info}")
    return (X_tr, y_tr, meta_tr), (X_val, y_val, meta_val), (X_te, y_te, meta_te), split_info


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "rmse": None,
            "mae": None,
            "smape": None,
            "directional_accuracy": None,
            "n_samples": 0,
        }

    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(np.mean(2.0 * np.abs(err) / denom))
    dir_acc = float((np.sign(y_pred) == np.sign(y_true)).mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "directional_accuracy": dir_acc,
        "n_samples": int(len(y_true)),
    }

# saves model under <prefix>/<run_id>/model.joblib
# With MODEL_PREFIX = 'models/xgboost', this becomes:
    #  models/xgboost/<run_id>/model.joblib
    # which matches the forecast script's expectations.
def save_model_to_s3(model, bucket: str, prefix: str, run_id: str) -> str:
    out_key = f"{prefix}/{run_id}/model.joblib"
    buf = io.BytesIO()
    dump(model, buf)
    buf.seek(0)
    S3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )
    log.info(f"Saved model to s3://{bucket}/{out_key}")
    return out_key


def save_metrics_to_s3(metrics: dict, bucket: str, prefix: str, run_id: str) -> str:
    """
    Save metrics under: <prefix>/<run_id>/metrics.json
    """
    out_key = f"{prefix}/{run_id}/metrics.json"
    S3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=json.dumps(metrics, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    log.info(f"[TRAIN] Saved metrics to s3://{bucket}/{out_key}")
    return out_key


# main file to parse cli args

def parse_args():
    parser = argparse.ArgumentParser()

    # S3 + run config
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--features-key", type=str, default="features/train.parquet")
    parser.add_argument("--model-prefix", type=str, default="models")
    parser.add_argument("--run-id", type=str, default="")

    # Split fractions
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)

    # XGBoost hyperparams
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)

    return parser.parse_args()


def main():
    args = parse_args()

    bucket = args.bucket
    features_key = args.features_key
    model_prefix = args.model_prefix
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    log.info(f"[TRAIN] Starting training run_id={run_id}")

    t_total_start = time.perf_counter()

    # 1) Load data
    t0 = time.perf_counter()
    df = load_features_from_s3(bucket, features_key)
    t_load = time.perf_counter() - t0
    log.info(f"[TRAIN] Loaded {len(df)} rows of features in {t_load:.3f}s")

    # 2) Prepare features/targets
    X, y, meta, feature_names = prepare_data(df)

    # 3) Time-based split
    (X_tr, y_tr, meta_tr), (X_val, y_val, meta_val), (X_te, y_te, meta_te), split_info = time_based_split(
        X, y, meta, train_frac=args.train_frac, val_frac=args.val_frac
    )

    # 4) Baseline
    baseline_metrics = {}
    if len(y_val) > 0:
        baseline_val = eval_metrics(y_val, np.zeros_like(y_val))
    else:
        baseline_val = None
    baseline_test = eval_metrics(y_te, np.zeros_like(y_te))
    baseline_metrics["zero"] = {"val": baseline_val, "test": baseline_test}
    log.info(f"[TRAIN] Baseline (zero) test metrics: {baseline_test}")

    # 5) Train XGBoost
    t_train_start = time.perf_counter()
    model = XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_lr,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=20 if len(y_val) > 0 else None,
    )

    if len(y_val) > 0:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_tr, y_tr, verbose=False)

    t_train = time.perf_counter() - t_train_start
    log.info(f"[TRAIN] Trained XGBoost in {t_train:.3f}s")

    # 6) Evaluate model
    t_eval_start = time.perf_counter()
    y_tr_pred = model.predict(X_tr)
    y_val_pred = model.predict(X_val) if len(y_val) > 0 else np.array([])
    y_te_pred = model.predict(X_te)
    t_eval = time.perf_counter() - t_eval_start

    metrics_xgb = {
        "train": eval_metrics(y_tr, y_tr_pred),
        "val": eval_metrics(y_val, y_val_pred) if len(y_val) > 0 else None,
        "test": eval_metrics(y_te, y_te_pred),
    }

    log.info(f"[TRAIN] XGBoost test metrics: {metrics_xgb['test']}")

    # 7) Save model + metrics to S3
    model_key = save_model_to_s3(model, bucket, model_prefix, run_id)
    t_total = time.perf_counter() - t_total_start

    metrics = {
        "run_id": run_id,
        "bucket": bucket,
        "features_key": features_key,
        "model_key": model_key,
        "feature_names": feature_names,
        "split_info": split_info,
        "baseline": baseline_metrics,
        "xgboost": metrics_xgb,
        "timing_seconds": {
            "load": t_load,
            "train": t_train,
            "eval": t_eval,
            "total": t_total,
        },
        "status": "success",
    }

    save_metrics_to_s3(metrics, bucket, model_prefix, run_id)
    log.info(f"[TRAIN] Done run_id={run_id}")


if __name__ == "__main__":
    main()
