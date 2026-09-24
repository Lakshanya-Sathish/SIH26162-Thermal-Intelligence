import pandas as pd
import numpy as np

FIRMS = "data/raw/firms_30day.csv"
PERSISTENCE = "data/processed/persistence_events.csv"
OUTPUT = "data/processed/features.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(FIRMS)
persistence = pd.read_csv(PERSISTENCE)

print("FIRMS rows:", len(df))
print("Persistence clusters:", len(persistence))

# --------------------------------------------------
# Confidence score
# --------------------------------------------------

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

# --------------------------------------------------
# Timestamp
# --------------------------------------------------

df["datetime"] = pd.to_datetime(
    df["acq_date"].astype(str)
    + " "
    + df["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
    errors="coerce"
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude",
        "datetime",
        "frp",
        "bright_ti4",
        "bright_ti5"
    ]
)

# --------------------------------------------------
# Spatial cluster matching
# --------------------------------------------------

cluster_coords = persistence[
    ["cluster_id", "latitude", "longitude"]
].values


def find_cluster(row):

    distances = (
        (cluster_coords[:, 1] - row["latitude"]) ** 2
        + (cluster_coords[:, 2] - row["longitude"]) ** 2
    )

    return cluster_coords[
        np.argmin(distances), 0
    ]


df["cluster_id"] = df.apply(
    find_cluster,
    axis=1
)

# --------------------------------------------------
# Persistence information
# --------------------------------------------------

df = df.merge(
    persistence[
        [
            "cluster_id",
            "detection_count",
            "days_detected",
            "avg_frp",
            "max_frp",
            "persistence"
        ]
    ],
    on="cluster_id",
    how="left"
)

# --------------------------------------------------
# Thermal features
# --------------------------------------------------

df["thermal_difference"] = (
    df["bright_ti4"] - df["bright_ti5"]
)

df["frp_log"] = np.log1p(df["frp"])

# --------------------------------------------------
# Temporal features
# --------------------------------------------------

df["hour"] = df["datetime"].dt.hour

df["day_of_week"] = (
    df["datetime"].dt.dayofweek
)

df["day_of_year"] = (
    df["datetime"].dt.dayofyear
)

# --------------------------------------------------
# Final feature table
# --------------------------------------------------

feature_columns = [
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "thermal_difference",
    "frp",
    "frp_log",
    "confidence_score",
    "hour",
    "day_of_week",
    "day_of_year",
    "detection_count",
    "days_detected",
    "avg_frp",
    "max_frp",
    "persistence"
]

features = df[feature_columns].copy()

# --------------------------------------------------
# Save
# --------------------------------------------------

features.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("FEATURE ENGINEERING COMPLETE")
print("==============================")

print("Rows:", len(features))
print("Columns:", len(features.columns))

print("\nFeatures:")
print(features.columns.tolist())

print("\nPersistence distribution:")
print(
    features["persistence"].value_counts()
)

print("\nSample:")
print(features.head())

print("\nSaved to:")
print(OUTPUT)