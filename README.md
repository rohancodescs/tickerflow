# TickerFlow - Event-Driven ETL & Stock Forecasting on AWS (with Local Baseline)

TickerFlow is an event-driven stock forecasting pipeline built on AWS. It ingests end-of-day (EOD) stock data, performs transformation + data quality (DQ) checks, builds ML features, trains an XGBoost model (weekly), generates next-trading-day forecasts (daily), and serves "forecast vs actual" analytics through Athena + a Streamlit dashboard.

This repository contains:

1. **AWS implementation** (serverless pipeline)
2. **Local baseline implementation** in a single Jupyter notebook for baseline vs cloud comparison

---

## Repository layout

**Cloud / AWS pipeline code**

- 'lambdas/ingest/handler.py' - ingestion Lambda (Alpha Vantage → 'raw/' in S3)
- 'lambdas/transform_features/handler.py' + 'transform_dq.py' - transform + DQ + feature build (writes 'processed/' and 'features/')
- 'lambda_forecast_xgboost/' - containerized forecast Lambda ("tickerflow-latest")
- 'lambda_train_sm/' - containerized training launcher Lambda ("tickerflow-weekly-train") + SageMaker training script

**Baseline**

- 'Baseline/baseline.ipynb' - single notebook that reproduces ingestion → transform/DQ → feature build → train → forecast locally
- 'Baseline/local_baseline_data/' - generated Parquet outputs

**Output**

- 'dashboard.py' - Streamlit dashboard that queries Athena view 'tickerflow_forecast_vs_actual'. run via "streamlit run dashboard.py after doing aws configure

**Not used**

- 'archive_not_used/' - older scripts kept for reference (not required to reproduce final results, should be in .gitignore)

---

## How the data is stored in S3 (AWS version)

TickerFlow uses an S3 "data lake" style layout:

- 'raw/<SYMBOL>/dt=<YYYY-MM-DD>/ingest.json' (normalized daily rows)
- 'processed/dt=<YYYY-MM-DD>/ohlcv.parquet' (clean OHLCV (open high low close volume) partitions)
- 'features/train.parquet' (model features)
- 'models/xgboost/<run_id>/{model.joblib, metrics.json}'
- 'forecasts/dt=<YYYY-MM-DD>/forecasts.parquet' (daily forecast partition)

The forecast Lambda uses "next trading day" logic (not naïvely '+1 day'), e.g., Friday → Monday.

---

## Reproducing the AWS pipeline (TA / grader workflow)

### Prereqs

- AWS Console access to the project account (IAM user sent to TA's personal email and Professor's email)
- Region: **us-east-1**
- Permissions to view **Lambda, S3, CloudWatch Logs, Athena, Glue, SageMaker**

### Step A - Run the pipeline (manual demo mode)

1. **Run ingestion Lambda:** 'tickerflow-ingest'
   - Expected effect: new objects appear under 's3://<bucket>/raw/<SYMBOL>/dt=<date>/ingest.json'

2. **Run transform + DQ + features:** 'tickerflow-transform-features'
   - Expected effect: new partitions under 'processed/dt=.../ohlcv.parquet' and updated 'features/train.parquet'

3. **Run forecasting Lambda:** 'tickerflow-latest'
   - Expected effect: 'forecasts/dt=<next_trading_day>/forecasts.parquet'

4. *(Optional)* **Run weekly training:** 'tickerflow-weekly-train'
   - Expected effect: SageMaker training job launches, writes 'models/xgboost/<run_id>/...'

### Step B - Refresh Athena partitions

In Athena, run (order doesn't matter):
'''sql
MSCK REPAIR TABLE tickerflow_prices;
MSCK REPAIR TABLE tickerflow_forecasts;
'''

Then verify:
'''sql
SELECT *
FROM tickerflow_forecast_vs_actual
ORDER BY target_date DESC, symbol
LIMIT 50;
'''

> **Note:** 'tickerflow_forecast_vs_actual' is a view joining 'tickerflow_prices' and 'tickerflow_forecasts' on '(symbol, target_date)' and exposes the fields used by the dashboard (forecast price, actual price, errors, etc.).

### Step C - View logs / metrics (CloudWatch)

For each Lambda invocation, CloudWatch's REPORT line provides:

- Duration + billed duration
- Memory size + **max memory used**
- Init duration (cold start), if applicable

---

## Running the Streamlit dashboard locally (reading AWS Athena)

### Prereqs (local machine)

- Python environment with dependencies installed (see 'requirements.txt')
- AWS credentials configured locally for the TA IAM user (via AWS CLI profile or env vars)
- Athena query result location must be configured (workgroup output or managed results)

### Steps

1. Create a venv and install deps:
'''bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
'''

2. Configure AWS credentials (one-time):
'''bash
aws configure --profile tickerflow-ta
# set region to us-east-1
'''

3. Run Streamlit:
'''bash
export AWS_PROFILE=tickerflow-ta
streamlit run dashboard.py
'''

---

## Reproducing the Local Baseline (Notebook)

This baseline runs the *same logical phases* locally: ingest → transform/DQ → features → train → forecast, and prints timing + accuracy metrics.

1. Open and run:
   - 'Baseline/baseline.ipynb'

2. The notebook pulls directly from Alpha Vantage, set:
   - 'ALPHAVANTAGE_KEY' in your environment before running the notebook (see the final report submission on ELMS)

3. Outputs (example):
   - 'Baseline/local_baseline_data/raw_prices_nov.parquet'
   - 'Baseline/local_baseline_data/features_nov.parquet'
   - 'Baseline/local_baseline_data/forecasts_nov.parquet'

---

## Troubleshooting (below are some issues that should be fixed but in the event they come up because you aren't the root user):

**Athena "No output location provided"**
Ensure the Athena workgroup has an output S3 location set or "managed query results" enabled.

**New S3 partitions not showing in Athena**
Re-run 'MSCK REPAIR TABLE ...' and confirm the S3 partition folder exists.

**Streamlit shows old data**
Clear Streamlit cache or restart the app.