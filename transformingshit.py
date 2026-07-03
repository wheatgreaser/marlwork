import pandas as pd
df = pd.read_parquet('./nyctaxi.parquet')
lookup = pd.read_csv('./lookup.csv')
print(lookup.head())

