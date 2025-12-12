import pandas as pd
from pyathena import connect
import streamlit as st

REGION = "us-east-1"
S3_STAGING = "s3://tickerflow-data-us-east-1/athena-results/"
DATABASE = "default"

@st.cache_data(ttl=60)  # refresh data at most every 60 seconds
def load_data():
    conn = connect(region_name=REGION, s3_staging_dir=S3_STAGING, schema_name=DATABASE)

    # Only show the most recent model run (e.g., the one that produced target_date = 2025-12-01)
    sql = """
    WITH latest_run AS (
        SELECT max(model_run_id) AS model_run_id
        FROM tickerflow_forecast_vs_actual
    )
    SELECT
        f.symbol,
        f.as_of_date,
        f.target_date,
        f.as_of_adj_close,
        f.forecast_adj_close,
        f.actual_adj_close,
        f.error_abs,
        f.error_pct,
        f.model_run_id
    FROM tickerflow_forecast_vs_actual f
    JOIN latest_run lr
      ON f.model_run_id = lr.model_run_id
    ORDER BY f.target_date, f.symbol
    """
    return pd.read_sql(sql, conn)

df = load_data()

st.title("TickerFlow – Forecast vs Actual")

# Small hint for the panel
if not df.empty:
    latest_run_id = df["model_run_id"].iloc[0]
    st.caption(f"Showing latest model run: {latest_run_id}")

symbols = sorted(df["symbol"].unique())
sel = st.multiselect("Symbols", symbols, default=symbols)

filtered = df[df["symbol"].isin(sel)]

st.write("Raw data", filtered)

if not filtered.empty:
    # Filter out rows where both values are None (non-trading / no actual yet)
    chart_data = filtered[
        filtered["forecast_adj_close"].notna() | filtered["actual_adj_close"].notna()
    ]

    if not chart_data.empty:
        chart_df = (
            chart_data[
                ["target_date", "symbol", "forecast_adj_close", "actual_adj_close"]
            ]
            .melt(
                id_vars=["target_date", "symbol"],
                value_vars=["forecast_adj_close", "actual_adj_close"],
                var_name="series",
                value_name="price",
            )
        )

        pivot_df = chart_df.pivot_table(
            index="target_date",
            columns=["symbol", "series"],
            values="price",
        )

        # Flatten MultiIndex columns: ('AAPL', 'forecast_adj_close') -> 'AAPL_forecast_adj_close'
        pivot_df.columns = ["_".join(col) for col in pivot_df.columns]

        st.line_chart(pivot_df)


