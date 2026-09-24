import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

INPUT = "data/processed/ml_features.csv"
OUTPUT = "data/processed/enriched_features.csv"

df = pd.read_csv(INPUT)

print("Events:", len(df))

# --------------------------------------------------
# Coordinates
# --------------------------------------------------

coords = np.radians(
    df[["latitude", "longitude"]].values
)

tree = BallTree(
    coords,
    metric="haversine"
)

# 5 km radius
radius_km = 5
earth_radius_km = 6371

radius = radius_km / earth_radius_km

# --------------------------------------------------
# Find nearby events
# --------------------------------------------------

neighbors = tree.query_radius(
    coords,
    r=radius
)

local_density = []
local_recurring = []
local_persistent = []
local_avg_frp = []

for i, indices in enumerate(neighbors):

    # Remove itself
    indices = indices[indices != i]

    nearby = df.iloc[indices]

    local_density.append(
        len(nearby)
    )

    local_recurring.append(
        (
            nearby["persistence"]
            == "Recurring"
        ).sum()
    )

    local_persistent.append(
        (
            nearby["persistence"]
            == "Persistent"
        ).sum()
    )

    if len(nearby) > 0:
        local_avg_frp.append(
            nearby["frp"].mean()
        )
    else:
        local_avg_frp.append(
            np.nan
        )

# --------------------------------------------------
# Add features
# --------------------------------------------------

df["nearby_detections_5km"] = local_density

df["nearby_recurring_5km"] = local_recurring

df["nearby_persistent_5km"] = local_persistent

df["nearby_avg_frp"] = local_avg_frp

# --------------------------------------------------
# Local activity classification
# --------------------------------------------------

def activity(row):

    if row["nearby_persistent_5km"] >= 1:
        return "Persistent Local Activity"

    elif row["nearby_recurring_5km"] >= 2:
        return "Recurring Local Activity"

    elif row["nearby_detections_5km"] >= 5:
        return "High Local Thermal Density"

    else:
        return "Isolated Thermal Activity"


df["local_activity"] = df.apply(
    activity,
    axis=1
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("SPATIAL CONTEXT COMPLETE")
print("==============================")

print(
    "Saved:",
    OUTPUT
)

print("\nLocal activity:")
print(
    df["local_activity"].value_counts()
)

print("\nTop spatially active events:")

print(
    df.nlargest(
        10,
        "nearby_detections_5km"
    )[
        [
            "latitude",
            "longitude",
            "frp",
            "persistence",
            "nearby_detections_5km",
            "nearby_recurring_5km",
            "nearby_persistent_5km",
            "local_activity"
        ]
    ]
)