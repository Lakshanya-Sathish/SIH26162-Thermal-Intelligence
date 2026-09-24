import pandas as pd

DATA_PATH = "data/raw/firms.csv"

df = pd.read_csv(DATA_PATH)

print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())