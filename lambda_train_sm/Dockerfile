# Dockerfile
FROM public.ecr.aws/lambda/python:3.13

# Install orchestration libraries
RUN pip install --upgrade pip && \
    pip install "sagemaker>=2.0,<3.0" boto3 --no-cache-dir

# Copy our Lambda handler + training script + requirements
COPY handler.py ${LAMBDA_TASK_ROOT}
COPY sm_train_xgboost.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Lambda entrypoint
CMD ["handler.lambda_handler"]
