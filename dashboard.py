import pandas as pd
from pyathena import connect
import streamlit as st

REGION = "us-east-1"
S3_STAGING = "s3://tickerflow-data-us-east-1/athena-results/"
DATABASE = "default"

@st.cache_data
def load_data():
    conn = connect(region_name=REGION, s3_staging_dir=S3_STAGING, schema_name=DATABASE)
    sql = """
    SELECT symbol,
           as_of_date,
           target_date,
           as_of_adj_close,
           forecast_adj_close,
           actual_adj_close,
           error_abs,
           error_pct,
           model_run_id
    FROM tickerflow_forecast_vs_actual
    ORDER BY target_date, symbol
    """
    return pd.read_sql(sql, conn)

df = load_data()

st.title("TickerFlow – Forecast vs Actual")

symbols = sorted(df["symbol"].unique())
sel = st.multiselect("Symbols", symbols, default=symbols)

filtered = df[df["symbol"].isin(sel)]

st.write("Raw data", filtered)

if not filtered.empty:
    # Basic line plot
    chart_df = (
        filtered[["target_date", "symbol", "forecast_adj_close", "actual_adj_close"]]
        .melt(id_vars=["target_date", "symbol"],
              value_vars=["forecast_adj_close", "actual_adj_close"],
              var_name="series",
              value_name="price")
    )
    st.line_chart(chart_df.pivot_table(index="target_date",
                                       columns=["symbol", "series"],
                                       values="price"))
