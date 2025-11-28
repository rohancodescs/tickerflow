# launch_sm_training.py
import time

import sagemaker
from sagemaker.sklearn import SKLearn

ROLE_ARN = "arn:aws:iam::312018064034:role/SageMakerExecutionRole-Tickerflow"
BUCKET = "tickerflow-data-us-east-1"
FEATURES_KEY = "features/train.parquet"
MODEL_PREFIX = "models"


def main():
    session = sagemaker.Session()
    region = session.boto_region_name
    print(f"Using region: {region}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"tickerflow-xgb-{timestamp}"

    estimator = SKLearn(
        entry_point="sm_train_xgboost.py",  # your training script
        source_dir=".",                     # assume script is in repo root
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",          # sklearn container version
        py_version="py3",
        sagemaker_session=session,
        hyperparameters={
            "bucket": BUCKET,
            "features-key": FEATURES_KEY,
            "model-prefix": MODEL_PREFIX,
            "train-frac": 0.7,
            "val-frac": 0.15,
            "xgb-n-estimators": 300,
            "xgb-max-depth": 4,
            "xgb-lr": 0.05,
            "xgb-subsample": 0.8,
            "xgb-colsample": 0.8,
        },
    )
    train_input = {
        "train": f"s3://{BUCKET}/{FEATURES_KEY}"
    }

    print(f"Starting training job: {job_name}")
    estimator.fit(inputs=train_input, job_name=job_name, wait=True)

    print("Training complete.")
    print("Training job name:", estimator.latest_training_job.name)
    print("Model artifacts (SageMaker default):", estimator.model_data)


if __name__ == "__main__":
    main()
