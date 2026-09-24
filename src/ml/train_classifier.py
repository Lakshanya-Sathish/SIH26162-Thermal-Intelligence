import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from xgboost import XGBClassifier


# =========================
# LOAD DATA
# =========================

INPUT = "data/processed/labeled_features.csv"
OUTPUT = "data/processed/classified_features.csv"

df = pd.read_csv(INPUT)

print("\nDataset:", df.shape)
print("\nClass distribution:")
print(df["target"].value_counts())


# =========================
# FEATURES
# =========================

features = [
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
    "nearby_detections_5km",
    "nearby_recurring_5km",
    "nearby_persistent_5km",
    "nearby_avg_frp",
    "anomaly_score"
]

# Keep only columns that actually exist
features = [x for x in features if x in df.columns]

X = df[features].copy()
y = df["target"].copy()

# Replace missing/infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)


# =========================
# ENCODE TARGET
# =========================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nClasses:")
for i, name in enumerate(encoder.classes_):
    print(i, "=", name)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# =========================
# RANDOM FOREST
# =========================

print("\n==============================")
print("TRAINING RANDOM FOREST")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Results:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))


# =========================
# XGBOOST
# =========================

print("\n==============================")
print("TRAINING XGBOOST")
print("==============================")

xgb = XGBClassifier(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("\nXGBoost Results:")
print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))


# =========================
# CHOOSE XGBOOST FOR PIPELINE
# =========================

df["predicted_class"] = encoder.inverse_transform(
    xgb.predict(X)
)

probabilities = xgb.predict_proba(X)

df["class_probability"] = probabilities.max(axis=1)


# =========================
# SAVE RESULTS
# =========================

df.to_csv(OUTPUT, index=False)

joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(xgb, "models/xgboost_classifier.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("\n==============================")
print("DONE")
print("==============================")

print("\nSaved:")
print("→ data/processed/classified_features.csv")
print("→ models/random_forest.pkl")
print("→ models/xgboost_classifier.pkl")
print("→ models/label_encoder.pkl")

print("\nPredicted classes:")
print(df["predicted_class"].value_counts())