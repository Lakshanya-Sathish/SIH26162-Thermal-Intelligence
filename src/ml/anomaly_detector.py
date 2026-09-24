import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

INPUT = "data/processed/features.csv"
OUTPUT = "data/processed/ml_features.csv"

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_csv(INPUT)

print("Input rows:", len(df))

# --------------------------------------------------
# ML features
# --------------------------------------------------

features = [
    "bright_ti4",
    "bright_ti5",
    "thermal_difference",
    "frp",
    "frp_log",
    "confidence_score",
    "hour",
    "detection_count",
    "days_detected",
    "avg_frp",
    "max_frp"
]

X = df[features].copy()

# Fill missing values
X = X.fillna(X.median())

# --------------------------------------------------
# Scale
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42
)

model.fit(X_scaled)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

df["ml_prediction"] = model.predict(X_scaled)

df["anomaly_score"] = -model.score_samples(X_scaled)

df["ml_anomaly"] = (
    df["ml_prediction"] == -1
)

# --------------------------------------------------
# Human-readable label
# --------------------------------------------------

df["ml_status"] = df["ml_anomaly"].map({
    True: "ML Anomaly",
    False: "Normal Thermal Pattern"
})

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print("\n==============================")
print("ML ANOMALY DETECTION COMPLETE")
print("==============================")

print(
    "ML anomalies:",
    df["ml_anomaly"].sum()
)

print(
    "Normal patterns:",
    (~df["ml_anomaly"]).sum()
)

print("\nTop anomalies:")

print(
    df.nlargest(
        10,
        "anomaly_score"
    )[
        [
            "latitude",
            "longitude",
            "frp",
            "days_detected",
            "detection_count",
            "persistence",
            "anomaly_score"
        ]
    ]
)

print(
    "\nSaved to:",
    OUTPUT
)