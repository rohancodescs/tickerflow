# lambdas/transform_features/handler.py
import os
import logging

from transform_dq import run_transform_and_features

log = logging.getLogger()
log.setLevel(logging.INFO)

def handler(event, context):
    bucket = os.environ.get("BUCKET", "tickerflow-data-us-east-1")
    raw_prefix = os.environ.get("RAW_PREFIX", "raw")
    processed_prefix = os.environ.get("PROCESSED_PREFIX", "processed")
    features_key = os.environ.get("FEATURES_KEY", "features/train.parquet")

    log.info("Starting transform + feature build for bucket=%s", bucket)

    result = run_transform_and_features(
        bucket=bucket,
        raw_prefix=raw_prefix,
        processed_prefix=processed_prefix,
        features_key=features_key,
    )

    log.info("Transform + features completed: %s", result)

    # Step Functions will get this JSON back
    return {
        "status": "ok",
        "bucket": bucket,
        "raw_prefix": raw_prefix,
        "processed_prefix": processed_prefix,
        "features_key": features_key,
        "metrics": result,
    }
