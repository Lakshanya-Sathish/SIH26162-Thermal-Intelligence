import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np

INPUT = "data/processed/features.csv"
OUTPUT = "data/processed/features_gis.csv"

# --------------------------------------------------
# Load thermal events
# --------------------------------------------------

df = pd.read_csv(INPUT)

print("Thermal events:", len(df))

# --------------------------------------------------
# Bounding box around our data
# --------------------------------------------------

west = df["longitude"].min() - 0.1
east = df["longitude"].max() + 0.1
south = df["latitude"].min() - 0.1
north = df["latitude"].max() + 0.1

bbox = (west, south, east, north)

print("\nQuerying OpenStreetMap...")
print("Bounding box:", bbox)

# --------------------------------------------------
# Download industrial areas
# --------------------------------------------------

tags = {
    "landuse": "industrial"
}

industrial = ox.features_from_bbox(
    bbox=bbox,
    tags=tags
)

print("OSM industrial features:", len(industrial))

# Remove features without geometry
industrial = industrial[
    industrial.geometry.notna()
].copy()

# --------------------------------------------------
# Convert thermal events to GeoDataFrame
# --------------------------------------------------

events = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)

# --------------------------------------------------
# Project to metric coordinate system
# --------------------------------------------------

events = events.to_crs("EPSG:32644")
industrial = industrial.to_crs("EPSG:32644")

# --------------------------------------------------
# Create representative point for industrial polygons
# --------------------------------------------------

industrial_points = industrial.copy()

industrial_points["geometry"] = (
    industrial_points.geometry.representative_point()
)

# --------------------------------------------------
# Find nearest industrial area
# --------------------------------------------------

nearest = gpd.sjoin_nearest(
    events,
    industrial_points[
        ["geometry"]
    ],
    how="left",
    distance_col="distance_to_industry_m"
)

# Convert metres → kilometres
nearest["distance_to_industry_km"] = (
    nearest["distance_to_industry_m"] / 1000
)

# Remove spatial join index
if "index_right" in nearest.columns:
    nearest = nearest.drop(
        columns=["index_right"]
    )

# Remove geometry before CSV
nearest = pd.DataFrame(nearest.drop(
    columns=["geometry"]
))

# Save
nearest.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("GIS ENRICHMENT COMPLETE")
print("==============================")

print(
    "Events enriched:",
    len(nearest)
)

print(
    "\nDistance statistics:"
)

print(
    nearest[
        "distance_to_industry_km"
    ].describe()
)

print(
    "\nSaved to:",
    OUTPUT
)