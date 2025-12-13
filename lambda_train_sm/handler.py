import os
import json
import logging
from datetime import datetime, timezone

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.exceptions import UnexpectedStatusException

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# helpers 

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y")


def lambda_handler(event, context):
    # Region / sessions
    region = boto3.Session().region_name or "us-east-1"
    boto_sess = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_sess)

    # Config from env
    bucket = os.environ.get("BUCKET", "tickerflow-data-us-east-1")
    features_key = os.environ.get("FEATURES_KEY", "features/train.parquet")
    model_prefix = os.environ.get("MODEL_PREFIX", "models/xgboost")
    training_role_arn = os.environ["SM_TRAINING_ROLE_ARN"]
    wait_for_completion = _env_bool("WAIT_FOR_COMPLETION", default=False)

    job_name = f"tickerflow-xgb-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    log.info("[TRAIN-LAMBDA] Starting SageMaker training job: %s", job_name)
    log.info("[TRAIN-LAMBDA] Using bucket=%s, features_key=%s", bucket, features_key)
    log.info("[TRAIN-LAMBDA] Using model_prefix=%s", model_prefix)
    log.info("[TRAIN-LAMBDA] Training role ARN=%s", training_role_arn)
    log.info("[TRAIN-LAMBDA] Wait for completion=%s", wait_for_completion)

    # This file lives in /var/task inside the Lambda image
    script_path = "sm_train_xgboost.py"

    # source_dir="." so the bundle includes requirements.txt
    estimator = SKLearn(
        entry_point=script_path,
        source_dir=".",  # bundle /var/task (handler + sm_train_xgboost + requirements.txt)
        role=training_role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="tickerflow-xgb",
        sagemaker_session=sm_session,
        hyperparameters={
            "bucket": bucket,
            "features-key": features_key,
            "model-prefix": model_prefix,
        },
    )

    train_input = {
        "train": TrainingInput(
            f"s3://{bucket}/{features_key}",
            content_type="application/x-parquet",
        )
    }

    try:
        estimator.fit(inputs=train_input, job_name=job_name, wait=wait_for_completion)
    except UnexpectedStatusException as e:
        log.error("[TRAIN-LAMBDA] Training job failed: %s", e, exc_info=True)
        return {
            "status": "failed",
            "job_name": job_name,
            "message": str(e),
        }

    # If we’re not waiting, just report submission
    if not wait_for_completion:
        log.info("[TRAIN-LAMBDA] Submitted training job %s (async).", job_name)
        return {
            "status": "submitted",
            "job_name": job_name,
            "training_role": training_role_arn,
        }

    # If we *did* wait and didn’t get an exception, job reached a terminal state.
    job_details = estimator.latest_training_job.describe()
    model_uri = job_details.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    training_status = job_details.get("TrainingJobStatus")

    log.info(
        "[TRAIN-LAMBDA] Job %s finished with status=%s, model_uri=%s",
        job_name,
        training_status,
        model_uri,
    )

    return {
        "status": "completed",
        "job_name": job_name,
        "training_job_status": training_status,
        "model_data": model_uri, 
    }
