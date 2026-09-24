import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

INPUT = "data/raw/firms_30day.csv"
OUTPUT = "data/processed/persistence_events.csv"

# Load data
df = pd.read_csv(INPUT)

# Combine date + time
df["datetime"] = pd.to_datetime(
    df["acq_date"].astype(str)
    + " "
    + df["acq_time"].astype(str).str.zfill(4),
    format="%Y-%m-%d %H%M",
    errors="coerce"
)

df = df.dropna(subset=["latitude", "longitude", "datetime"])

# --------------------------------------------------
# 1. Spatial clustering
# --------------------------------------------------

coords = np.radians(
    df[["latitude", "longitude"]].values
)

# ~1 km spatial radius
EARTH_RADIUS_KM = 6371.0
RADIUS_KM = 1.0

eps = RADIUS_KM / EARTH_RADIUS_KM

clusterer = DBSCAN(
    eps=eps,
    min_samples=1,
    metric="haversine"
)

df["cluster_id"] = clusterer.fit_predict(coords)

# --------------------------------------------------
# 2. Calculate persistence
# --------------------------------------------------

summary = (
    df.groupby("cluster_id")
    .agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        first_seen=("datetime", "min"),
        last_seen=("datetime", "max"),
        detection_count=("datetime", "count"),
        avg_frp=("frp", "mean"),
        max_frp=("frp", "max"),
    )
    .reset_index()
)

# Number of unique days
days_seen = (
    df.groupby("cluster_id")["datetime"]
    .apply(lambda x: x.dt.date.nunique())
    .reset_index(name="days_detected")
)

summary = summary.merge(
    days_seen,
    on="cluster_id"
)

# --------------------------------------------------
# 3. Persistence classification
# --------------------------------------------------

def classify_persistence(row):

    days = row["days_detected"]
    detections = row["detection_count"]

    if days >= 15 or detections >= 20:
        return "Persistent"

    elif days >= 5 or detections >= 5:
        return "Recurring"

    else:
        return "Transient"


summary["persistence"] = summary.apply(
    classify_persistence,
    axis=1
)

# --------------------------------------------------
# Save
# --------------------------------------------------

summary.to_csv(OUTPUT, index=False)

print("\n==============================")
print("PERSISTENCE ENGINE COMPLETE")
print("==============================")

print("Clusters:", len(summary))

print("\nPersistence distribution:")
print(summary["persistence"].value_counts())

print("\nSample:")
print(summary.head(10))

print("\nSaved to:")
print(OUTPUT)