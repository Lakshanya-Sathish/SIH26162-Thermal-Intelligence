import pandas as pd

INPUT = "data/raw/firms.csv"
OUTPUT = "data/processed/clean_firms.csv"

df = pd.read_csv(INPUT)

# Keep the important columns
df = df[
    [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "confidence",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "daynight",
    ]
].copy()

# Convert numeric columns
numeric_cols = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "frp",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert confidence to numeric score
confidence_map = {
    "l": 0.33,
    "n": 0.66,
    "h": 1.00
}

df["confidence_score"] = (
    df["confidence"]
    .astype(str)
    .str.lower()
    .map(confidence_map)
)

# Combine date + time
df["datetime"] = pd.to_datetime(
    df["acq_date"].astype(str)
    + " "
    + df["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
    errors="coerce"
)

# Remove invalid rows
df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "bright_ti4",
        "frp",
        "datetime"
    ]
)

# Sort chronologically
df = df.sort_values("datetime").reset_index(drop=True)

# Save
df.to_csv(OUTPUT, index=False)

print("Cleaning complete!")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nSample:")
print(df.head())
print("\nSaved to:", OUTPUT)