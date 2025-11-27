import pandas as pd
pd.set_option('display.max_columns', None) # To display all columns
df = pd.read_parquet("test.parquet")
print(df.head(10))
