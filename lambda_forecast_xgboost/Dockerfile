FROM public.ecr.aws/lambda/python:3.13

RUN pip install --upgrade pip && \
    pip install \
        pandas==2.2.3 \
        pyarrow==18.1.0 \
        xgboost==2.1.4 \
        joblib==1.4.2 \
        --no-cache-dir

COPY handler.py ${LAMBDA_TASK_ROOT}

CMD ["handler.handler"]