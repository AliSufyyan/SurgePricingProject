"""
Surge Pricing Dashboard — Streamlit UI
NYC Taxi Big Data Project

Run:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚕 Surge Pricing Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SURGE_COLORS = {
    1.0: "#27AE60", 1.4: "#F1C40F", 1.6: "#E67E22",
    2.0: "#E74C3C", 2.5: "#C0392B", 2.8: "#8E44AD",
    3.0: "#6C3483", 3.2: "#1A252F",
}
DAY_MAP = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
DAY_ORDER = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
PALETTE = "#2C3E50"
ACCENT  = "#E74C3C"
ACCENT2 = "#3498DB"
BG      = "#F8F9FA"

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_viz_data.csv")
    pred = pd.read_csv(
        "data/predictions_sample/part-00000-95f4922d-fde0-46d2-8d7e-0bdaefd36c65-c000.csv"
    )
    df["day_name"] = pd.Categorical(
        df["day_of_week"].map(DAY_MAP), categories=DAY_ORDER, ordered=True
    )
    return df, pred

df, pred = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/taxi.png", width=80)
st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("---")

day_filter = st.sidebar.multiselect(
    "Day of Week", options=DAY_ORDER, default=DAY_ORDER
)
hour_range = st.sidebar.slider(
    "Hour Range", min_value=0, max_value=23, value=(0, 23)
)
surge_filter = st.sidebar.multiselect(
    "Surge Multipliers", options=sorted(df["surge_multiplier"].unique()),
    default=sorted(df["surge_multiplier"].unique())
)
day_type = st.sidebar.radio(
    "Day Type", options=["All", "Weekday Only", "Weekend Only"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Info**")
st.sidebar.markdown("- Model: Random Forest (Spark MLlib)")
st.sidebar.markdown("- Dataset: NYC Yellow Taxi 2016")
st.sidebar.markdown("- Pipeline: Kafka → Spark → Dashboard")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────
filtered = df[
    (df["day_name"].isin(day_filter)) &
    (df["hour"].between(hour_range[0], hour_range[1])) &
    (df["surge_multiplier"].isin(surge_filter))
]
if day_type == "Weekday Only":
    filtered = filtered[filtered["is_weekend"] == 0]
elif day_type == "Weekend Only":
    filtered = filtered[filtered["is_weekend"] == 1]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚕 NYC Taxi — Surge Pricing Analytics")
st.markdown("**Big Data Pipeline: Kafka → PySpark Batch & Streaming → ML (Random Forest) → Dashboard**")
st.markdown("---")

if len(filtered) == 0:
    st.warning("⚠️ No data matches your filters. Please adjust the sidebar.")
    st.stop()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

avg_surge = filtered["surge_multiplier"].mean()
max_surge = filtered["surge_multiplier"].max()
total_trips = len(filtered)
avg_fare = filtered["fare_amount"].mean()
avg_dist = filtered["trip_distance"].mean()

residuals = pred["label"] - pred["prediction"]
model_rmse = np.sqrt((residuals**2).mean())

k1.metric("📊 Avg Surge",     f"{avg_surge:.2f}x")
k2.metric("🔥 Peak Surge",    f"{max_surge:.1f}x")
k3.metric("🚗 Total Trips",   f"{total_trips:,}")
k4.metric("💰 Avg Fare",      f"${avg_fare:.2f}")
k5.metric("🎯 Model RMSE",    f"{model_rmse:.4f}")

st.markdown("---")

# ── ROW 1: CHART 1 + CHART 2 ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

# Chart 1 — Hourly Surge Bar Chart
with col1:
    st.subheader("⏰ Average Surge by Hour")
    hourly = filtered.groupby("hour")["surge_multiplier"].mean().reindex(range(24), fill_value=np.nan)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [ACCENT if v >= 2.5 else ACCENT2 if v >= 1.5 else "#27AE60"
              for v in hourly.fillna(0)]
    bars = ax.bar(hourly.index, hourly.values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(filtered["surge_multiplier"].mean(), color=PALETTE, linestyle="--",
               linewidth=1.2, label=f"Avg: {filtered['surge_multiplier'].mean():.2f}x")
    ax.set_xlabel("Hour of Day", fontsize=11)
    ax.set_ylabel("Avg Surge Multiplier", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.set_ylim(0, 3.8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Chart 2 — Day of Week Box Plot
with col2:
    st.subheader("📅 Surge Distribution by Day")
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.set_facecolor(BG)
    day_palette = [ACCENT if d in ["Sat","Sun"] else ACCENT2 for d in DAY_ORDER]
    present_days = [d for d in DAY_ORDER if d in filtered["day_name"].values]
    if present_days:
        sns.boxplot(
            data=filtered, x="day_name", y="surge_multiplier",
            order=present_days,
            palette=[ACCENT if d in ["Sat","Sun"] else ACCENT2 for d in present_days],
            ax=ax, linewidth=1.2,
            flierprops=dict(marker=".", markersize=2, alpha=0.3)
        )
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Surge Multiplier", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    wknd = mpatches.Patch(color=ACCENT,  label="Weekend")
    wkdy = mpatches.Patch(color=ACCENT2, label="Weekday")
    ax.legend(handles=[wknd, wkdy], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── ROW 2: CHART 3 + CHART 4 ─────────────────────────────────────────────────
col3, col4 = st.columns(2)

# Chart 3 — Donut Chart
with col3:
    st.subheader("🍩 Trips by Surge Level")
    surge_counts = filtered["surge_multiplier"].value_counts().sort_index()
    colors_pie = [SURGE_COLORS.get(k, "#95A5A6") for k in surge_counts.index]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)
    wedges, texts, autotexts = ax.pie(
        surge_counts, labels=[f"{k}x" for k in surge_counts.index],
        autopct="%1.1f%%", colors=colors_pie, startangle=90,
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
        pctdistance=0.75, textprops=dict(fontsize=9)
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_fontweight("bold"); at.set_color("white")
    centre = plt.Circle((0,0), 0.40, color=BG)
    ax.add_patch(centre)
    ax.text(0, 0, f"{len(filtered):,}\ntrips", ha="center", va="center",
            fontsize=10, fontweight="bold", color=PALETTE)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Chart 4 — Actual vs Predicted
with col4:
    st.subheader("🎯 Actual vs Predicted Surge")
    sample_pred = pred.sample(n=min(2000, len(pred)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(sample_pred["label"], sample_pred["prediction"],
               alpha=0.3, s=15, color=ACCENT2, edgecolors="none")
    lims = [
        min(sample_pred["label"].min(), sample_pred["prediction"].min()) - 0.05,
        max(sample_pred["label"].max(), sample_pred["prediction"].max()) + 0.05
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    r = sample_pred["label"] - sample_pred["prediction"]
    ax.set_xlabel("Actual Surge", fontsize=11)
    ax.set_ylabel("Predicted Surge", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.92,
            f"RMSE: {np.sqrt((r**2).mean()):.4f}\nMAE:  {r.abs().mean():.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7"))
    ax.grid(alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── ROW 3: HEATMAP + FEATURE IMPORTANCE ──────────────────────────────────────
col5, col6 = st.columns(2)

# Chart 5 — Heatmap
with col5:
    st.subheader("🔥 Surge Heatmap: Hour × Day")
    pivot_data = filtered.pivot_table(
        values="surge_multiplier", index="day_of_week", columns="hour", aggfunc="mean"
    )
    pivot_data.index = [DAY_MAP[i] for i in pivot_data.index]
    # Reorder to day order
    present = [d for d in DAY_ORDER if d in pivot_data.index]
    pivot_data = pivot_data.reindex(present)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(pivot_data, ax=ax, cmap="YlOrRd", linewidths=0.3,
                linecolor="white", annot=True, fmt=".1f",
                annot_kws={"size": 8}, cbar_kws={"label": "Avg Surge"})
    ax.set_xlabel("Hour", fontsize=11)
    ax.set_ylabel("Day", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Chart 6 — Feature Importances
with col6:
    st.subheader("🌲 Feature Importances (Random Forest)")
    feature_names = [
        "hour", "day_of_week", "month", "is_weekend",
        "trip_distance", "passenger_count", "trip_duration_min", "avg_speed_mph",
        "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "is_morning_peak", "is_evening_peak", "is_late_night",
    ]
    importances = [0.3124, 0.1856, 0.0045, 0.0981,
                   0.0612, 0.0321, 0.0583, 0.1243,
                   0.0089, 0.0095, 0.0092, 0.0087,
                   0.0201, 0.0198, 0.0473]
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1])
    names, vals = zip(*fi)
    colors_fi = [ACCENT if v >= 0.15 else ACCENT2 if v >= 0.05 else "#95A5A6" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.barh(names, vals, color=colors_fi, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(axis="x", alpha=0.4)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── ROW 4: FULL-WIDTH CHARTS ──────────────────────────────────────────────────
st.subheader("🌙 Weekday vs Weekend Surge by Hour")
wkdy = filtered[filtered["is_weekend"] == 0].groupby("hour")["surge_multiplier"].mean()
wknd = filtered[filtered["is_weekend"] == 1].groupby("hour")["surge_multiplier"].mean()

fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG)
ax.set_facecolor(BG)
if not wkdy.empty:
    ax.plot(wkdy.index, wkdy.values, color=ACCENT2, linewidth=2.5,
            marker="o", markersize=6, label="Weekday")
if not wknd.empty:
    ax.plot(wknd.index, wknd.values, color=ACCENT, linewidth=2.5,
            marker="s", markersize=6, label="Weekend")
ax.axvspan(7,  9,  alpha=0.08, color=ACCENT2)
ax.axvspan(17, 19, alpha=0.08, color=ACCENT)
ax.axvspan(22, 24, alpha=0.08, color="#8E44AD")
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Avg Surge Multiplier", fontsize=11)
ax.set_xticks(range(0, 24))
ax.legend(fontsize=10)
ax.grid(alpha=0.4)
ax.set_ylim(0, 3.8)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── RAW DATA TABLE ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 View Filtered Raw Data"):
    show_cols = ["hour", "day_name", "is_weekend", "trip_distance",
                 "fare_amount", "passenger_count", "avg_speed_mph",
                 "trip_duration_min", "surge_multiplier"]
    st.dataframe(
        filtered[show_cols].sample(n=min(500, len(filtered)), random_state=42)
            .sort_values("surge_multiplier", ascending=False)
            .reset_index(drop=True),
        use_container_width=True
    )

with st.expander("📋 View Model Predictions Sample"):
    st.dataframe(pred.head(100), use_container_width=True)

st.markdown("---")
st.caption("Built with PySpark · Kafka · Streamlit · Matplotlib · Seaborn")
