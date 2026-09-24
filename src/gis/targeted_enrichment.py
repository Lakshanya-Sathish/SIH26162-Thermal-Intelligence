import pandas as pd
import numpy as np
import osmnx as ox
from math import radians, sin, cos, sqrt, atan2

INPUT = "data/processed/ml_features.csv"
OUTPUT = "data/processed/enriched_features.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT)

print("Loaded events:", len(df))

# --------------------------------------------------
# Select important events only
# --------------------------------------------------

important = df[
    (
        df["persistence"].isin(
            ["Recurring", "Persistent"]
        )
    )
    |
    (df["ml_anomaly"] == True)
].copy()

# Prioritize persistent/anomalous events
important = important.sort_values(
    by=[
        "days_detected",
        "anomaly_score"
    ],
    ascending=False
)

# Only enrich top 10
important = important.head(10)

print(
    "Events selected for GIS enrichment:",
    len(important)
)

# --------------------------------------------------
# Haversine distance
# --------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    return (
        2
        * R
        * atan2(
            sqrt(a),
            sqrt(1 - a)
        )
    )


# --------------------------------------------------
# Query categories
# --------------------------------------------------

categories = {
    "industrial": {
        "landuse": "industrial"
    },

    "forest": {
        "landuse": "forest"
    },

    "farmland": {
        "landuse": "farmland"
    },

    "mining": {
        "landuse": "quarry"
    }
}


# --------------------------------------------------
# Enrichment function
# --------------------------------------------------

def get_nearest_distance(
    lat,
    lon,
    tags
):

    try:

        gdf = ox.features_from_point(
            center_point=(lat, lon),
            tags=tags,
            dist=5000
        )

        if gdf.empty:
            return np.nan

        distances = []

        for geometry in gdf.geometry:

            if geometry is None:
                continue

            try:

                point = geometry.representative_point()

                d = haversine(
                    lat,
                    lon,
                    point.y,
                    point.x
                )

                distances.append(d)

            except Exception:
                continue

        if not distances:
            return np.nan

        return min(distances)

    except Exception as e:

        print(
            "OSM query failed:",
            str(e)[:100]
        )

        return np.nan


# --------------------------------------------------
# Run enrichment
# --------------------------------------------------

for category, tags in categories.items():

    print(
        f"\nSearching for {category}..."
    )

    distances = []

    for _, row in important.iterrows():

        distance = get_nearest_distance(
            row["latitude"],
            row["longitude"],
            tags
        )

        distances.append(distance)

        print(
            f"  {row['latitude']:.4f}, "
            f"{row['longitude']:.4f} "
            f"→ {distance}"
        )

    important[
        f"distance_to_{category}_km"
    ] = distances


# --------------------------------------------------
# Merge back into full dataset
# --------------------------------------------------

enriched = df.copy()

for column in [
    "distance_to_industrial_km",
    "distance_to_forest_km",
    "distance_to_farmland_km",
    "distance_to_mining_km"
]:

    enriched[column] = np.nan

for index, row in important.iterrows():

    for column in [
        "distance_to_industrial_km",
        "distance_to_forest_km",
        "distance_to_farmland_km",
        "distance_to_mining_km"
    ]:

        enriched.loc[
            index,
            column
        ] = row[column]


# --------------------------------------------------
# Save
# --------------------------------------------------

enriched.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("TARGETED GIS ENRICHMENT DONE")
print("==============================")

print("\nImportant events enriched:")

print(
    enriched.loc[
        important.index,
        [
            "latitude",
            "longitude",
            "persistence",
            "ml_anomaly",
            "distance_to_industrial_km",
            "distance_to_forest_km",
            "distance_to_farmland_km",
            "distance_to_mining_km"
        ]
    ]
)

print(
    "\nSaved to:",
    OUTPUT
)