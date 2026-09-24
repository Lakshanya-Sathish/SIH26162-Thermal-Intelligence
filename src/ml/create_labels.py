import pandas as pd
import numpy as np

INPUT = "data/processed/enriched_features.csv"
OUTPUT = "data/processed/labeled_features.csv"

df = pd.read_csv(INPUT)

# --------------------------------------------------
# Prototype weak-label generator
# --------------------------------------------------

def assign_label(row):

    persistence = row["persistence"]
    frp = row["frp"]
    days = row["days_detected"]
    density = row["nearby_detections_5km"]

    # Very persistent + strong thermal signature
    # → prototype gas flare candidate
    if (
        persistence == "Persistent"
        and days >= 7
        and frp >= df["frp"].quantile(0.75)
    ):
        return "Gas Flare"

    # High local thermal activity
    # → prototype industrial candidate
    elif (
        density >= 5
        and frp >= df["frp"].quantile(0.60)
    ):
        return "Industrial Fire"

    # Persistent / recurring isolated activity
    # → prototype mining/other
    elif (
        persistence == "Persistent"
        and density < 5
    ):
        return "Mining / Other"

    # Strong but short-lived thermal event
    elif (
        persistence == "Transient"
        and frp >= df["frp"].quantile(0.75)
    ):
        return "Wildfire"

    # Remaining recurring/transient events
    else:
        return "Agricultural Burning"


df["target"] = df.apply(
    assign_label,
    axis=1
)

df.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("PROTOTYPE LABELS CREATED")
print("==============================")

print(
    df["target"].value_counts()
)

print(
    "\nSaved to:",
    OUTPUT
)