import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# CONFIG
# ============================================================

DATA = "data/processed/classified_features.csv"
CLUSTER_DATA = "data/processed/persistence_events.csv"

st.set_page_config(
    page_title="SIH26162 | Thermal Intelligence",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA)
clusters = pd.read_csv(CLUSTER_DATA)


# ============================================================
# PROTOTYPE CLASSIFICATION
# ============================================================

frp_threshold = df["frp"].quantile(0.75)
brightness_threshold = df["bright_ti4"].quantile(0.75)


df["source_type"] = df["predicted_class"]


# ============================================================
# HEADER
# ============================================================

st.title("🔥 Thermal Intelligence Dashboard")

st.caption(
    "Satellite-based thermal anomaly detection, "
    "spatial-temporal persistence and explainable analysis"
)

st.info(
    "🤖 XGBoost classification uses thermal, temporal, persistence, "
    "anomaly and local spatial features."
)

st.caption(
    "Prototype classification trained using weakly supervised labels."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

persistence_options = [
    "All",
    "Transient",
    "Recurring",
    "Persistent"
]

selected_persistence = st.sidebar.selectbox(
    "Persistence",
    persistence_options
)

if selected_persistence == "All":

    filtered = df.copy()

else:

    filtered = df[
        df["persistence"] == selected_persistence
    ].copy()


source_options = [
    "All"
] + sorted(
    df["source_type"].unique().tolist()
)

selected_source = st.sidebar.selectbox(
    "Source Type",
    source_options
)

if selected_source != "All":

    filtered = filtered[
        filtered["source_type"] == selected_source
    ].copy()


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)


c1.metric(
    "Thermal Detections",
    f"{len(df):,}"
)


c2.metric(
    "Spatial Clusters",
    f"{len(clusters):,}"
)


c3.metric(
    "Recurring",
    f"{(df['persistence'] == 'Recurring').sum():,}"
)


c4.metric(
    "Persistent",
    f"{(df['persistence'] == 'Persistent').sum():,}"
)

c5.metric(
    "High-Intensity Persistent",
    f"{(
        df['source_type'] == 'High-Intensity Persistent'
    ).sum():,}"
)

c6.metric(
    "Sources Requiring Review",
    f"{(
        df['persistence'].isin(
            ['Recurring', 'Persistent']
        )
    ).sum():,}"
)

c7.metric(
    "ML Anomalies",
    f"{df['ml_anomaly'].sum():,}"
)

st.markdown("### 🤖 AI Source Classification")

class_counts = df["predicted_class"].value_counts().reset_index()
class_counts.columns = ["Source Type", "Detections"]

st.bar_chart(
    class_counts.set_index("Source Type")
)

# ============================================================
# MAP
# ============================================================

st.subheader("🗺️ Thermal Anomaly Map")

color_map = {
    "Gas Flare": "#e63946",
    "Industrial Fire": "#f4a261",
    "Wildfire": "#2a9d8f",
    "Agricultural Burning": "#e9c46a",
    "Mining / Other": "#457b9d"
}


if len(filtered) > 0:

    fig = px.scatter_map(
        filtered,
        lat="latitude",
        lon="longitude",
        color="source_type",
        color_discrete_map=color_map,
        size="frp",

        hover_name="source_type",

        hover_data={
            "latitude": ":.4f",
            "longitude": ":.4f",
            "frp": ":.2f",
            "bright_ti4": ":.2f",
            "confidence_score": ":.2f",
            "detection_count": True,
            "days_detected": True,
            "latitude": False,
            "longitude": False
        },

        zoom=5,
        height=600
    )

    fig.update_layout(
        map_style="open-street-map",
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        legend_title_text="Source Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:

    st.warning(
        "No events match the selected filters."
    )


# ============================================================
# EVENT ANALYSIS
# ============================================================

st.subheader("🔎 Event Analysis")


if len(filtered) > 0:

    # Show the most important events first
    priority_events = filtered[
        filtered["persistence"].isin(
            ["Recurring", "Persistent"]
        )
    ].copy()

    # If no recurring/persistent events exist,
    # fall back to all filtered events
    if len(priority_events) == 0:
        priority_events = filtered.copy()

    event_ids = priority_events.index.tolist()

    selected_index = st.selectbox(
        "Select a significant thermal source",
        event_ids,
        format_func=lambda x:
            f"Event {x} — "
            f"{priority_events.loc[x, 'source_type']} — "
            f"{int(priority_events.loc[x, 'days_detected'])} days detected"
    )
    event = priority_events.loc[selected_index]


    # --------------------------------------------------------
    # Event heading
    # --------------------------------------------------------

    st.markdown(
        f"### {event['source_type']}"
    )

    st.markdown("### 🤖 AI Classification")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Predicted Source",
            event["predicted_class"]
        )

    with c2:
        st.metric(
            "Model Probability",
            f"{event['class_probability'] * 100:.1f}%"
        )

    st.caption(
        "Classification generated by the XGBoost prototype using thermal, "
        "temporal, persistence, and local spatial features."
    )

    st.markdown("### 🔎 Explainable Evidence")

    evidence = []

    if event["frp"] >= df["frp"].quantile(0.75):
        evidence.append("High fire radiative power (FRP)")

    if event["days_detected"] >= 7:
        evidence.append(
            f"Persistent thermal activity ({int(event['days_detected'])} days detected)"
        )

    if event["nearby_detections_5km"] >= 5:
        evidence.append(
            f"High local thermal density ({int(event['nearby_detections_5km'])} nearby detections)"
        )

    if event["nearby_persistent_5km"] >= 1:
        evidence.append("Nearby persistent thermal activity")

    if event["ml_anomaly"] == -1:
        evidence.append("Detected as an unusual thermal pattern by anomaly detection")

    for item in evidence:
        st.write("•", item)


    # --------------------------------------------------------
    # Two columns
    # --------------------------------------------------------

    left, right = st.columns(2)


    # --------------------------------------------------------
    # Thermal profile
    # --------------------------------------------------------

    with left:

        st.markdown("#### 🌡️ Thermal Profile")

        st.write(
            f"**FRP:** "
            f"{event['frp']:.2f}"
        )

        st.write(
            f"**Brightness Temperature:** "
            f"{event['bright_ti4']:.2f} K"
        )

        st.write(
            f"**Brightness Difference:** "
            f"{event['thermal_difference']:.2f} K"
        )

        st.write(
            f"**Confidence Score:** "
            f"{event['confidence_score']:.2f}"
        )

        st.write(
            f"**Location:** "
            f"{event['latitude']:.4f}, "
            f"{event['longitude']:.4f}"
        )

        st.write(
            f"**ML Status:** "
            f"{event['ml_status']}"
        )       

        st.write(
            f"**Anomaly Score:** "
            f"{event['anomaly_score']:.3f}"
        )


    # --------------------------------------------------------
    # Persistence profile
    # --------------------------------------------------------

    with right:

        st.markdown("#### 🔁 Persistence Profile")

        st.write(
            f"**Persistence:** "
            f"{event['persistence']}"
        )

        st.write(
            f"**Detections:** "
            f"{int(event['detection_count'])}"
        )

        st.write(
            f"**Days Detected:** "
            f"{int(event['days_detected'])}"
        )

        st.write(
            f"**Average FRP:** "
            f"{event['avg_frp']:.2f}"
        )

        st.write(
            f"**Maximum FRP:** "
            f"{event['max_frp']:.2f}"
        )

        # --------------------------------------------------------
        # Spatial context
        # --------------------------------------------------------

        st.markdown("#### 📍 Spatial Context")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Nearby Detections",
            int(event["nearby_detections_5km"])
        )

        s2.metric(
            "Nearby Recurring",
            int(event["nearby_recurring_5km"])
        )

        s3.metric(
            "Nearby Persistent",
            int(event["nearby_persistent_5km"])
        )

        s4.metric(
            "Local Activity",
            event["local_activity"]
        )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown("---")

    st.markdown(
        "### 🧠 Why was this event flagged?"
    )

    reasons = []


    # High FRP

    if event["frp"] >= frp_threshold:

        reasons.append(
            f"High thermal radiative power "
            f"(FRP = {event['frp']:.2f})"
        )


    # High brightness

    if event["bright_ti4"] >= brightness_threshold:

        reasons.append(
            f"High brightness temperature "
            f"({event['bright_ti4']:.2f} K)"
        )


    # Repeated detections

    if event["days_detected"] >= 5:

        reasons.append(
            f"Detected across "
            f"{int(event['days_detected'])} "
            f"different days"
        )


    # Detection count

    if event["detection_count"] >= 5:

        reasons.append(
            f"{int(event['detection_count'])} "
            f"detections associated with this "
            f"spatial cluster"
        )

    if event["nearby_detections_5km"] >= 5:
        reasons.append(
            f"High local thermal density: "
            f"{int(event['nearby_detections_5km'])} "
            f"detections within 5 km"
        )

    if event["nearby_recurring_5km"] >= 1:
        reasons.append(
            f"{int(event['nearby_recurring_5km'])} "
            "recurring thermal source(s) nearby"
        )

    if event["nearby_persistent_5km"] >= 1:
        reasons.append(
            f"{int(event['nearby_persistent_5km'])} "
            "persistent thermal source nearby"
        )

    if event["ml_anomaly"]:
        reasons.append(
            "ML identified an unusual combination "
            "of thermal and temporal features"
        )

    # Persistent

    if event["persistence"] == "Persistent":

        reasons.append(
            "Long-term spatial-temporal persistence"
        )

    # Recurring

    elif event["persistence"] == "Recurring":

        reasons.append(
            "Repeated thermal activity over time"
        )

    if event["ml_anomaly"]:

        st.warning(
            "⚠ ML detected an unusual thermal feature pattern."
        )

    else:

        st.info(
            "ML pattern is within the normal range "
            "of the dataset."
        )


    # Display reasons

    if reasons:

        for reason in reasons:

            st.success(
                "✓ " + reason
            )

    else:

        st.info(
            "Thermal anomaly detected, "
            "but intensity and persistence "
            "are relatively low."
        )

st.markdown("#### 📍 Spatial Context")

st.write(
    f"**Nearby detections (5 km):** "
    f"{int(event['nearby_detections_5km'])}"
)

st.write(
    f"**Nearby recurring sources:** "
    f"{int(event['nearby_recurring_5km'])}"
)

st.write(
    f"**Nearby persistent sources:** "
    f"{int(event['nearby_persistent_5km'])}"
)

st.write(
    f"**Local activity:** "
    f"{event['local_activity']}"
)
# ============================================================
# PERSISTENCE DISTRIBUTION
# ============================================================

st.markdown("---")

st.subheader("📊 Persistence Distribution")


counts = (
    df["persistence"]
    .value_counts()
    .reset_index()
)

counts.columns = [
    "Persistence",
    "Count"
]


chart = px.bar(
    counts,
    x="Persistence",
    y="Count",
    title="Thermal Source Persistence"
)


chart.update_layout(
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    )
)


st.plotly_chart(
    chart,
    use_container_width=True
)


# ============================================================
# SOURCE TYPE DISTRIBUTION
# ============================================================

st.subheader("🔥 Prototype Source Classification")


source_counts = (
    df["source_type"]
    .value_counts()
    .reset_index()
)

source_counts.columns = [
    "Source Type",
    "Count"
]


source_chart = px.bar(
    source_counts,
    x="Source Type",
    y="Count",
    title="Thermal Source Categories"
)


source_chart.update_layout(
    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    )
)


st.plotly_chart(
    source_chart,
    use_container_width=True
)
st.subheader("📍 Local Thermal Activity")

activity_counts = (
    df["local_activity"]
    .value_counts()
    .reset_index()
)

activity_counts.columns = [
    "Activity",
    "Count"
]

activity_chart = px.bar(
    activity_counts,
    x="Activity",
    y="Count",
    title="Spatial Activity Around Thermal Events"
)

st.plotly_chart(
    activity_chart,
    use_container_width=True
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "SIH26162 • Prototype system • "
    "NASA FIRMS thermal anomaly data"
)