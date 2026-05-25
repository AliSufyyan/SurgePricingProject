"""
Surge Pricing — Matplotlib & Seaborn Visualizations
NYC Taxi Big Data Project
Run: python visualizations.py
Outputs: surge_visualizations.png (all 8 charts in one figure)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH        = "data/processed_viz_data.csv"
PREDICTIONS_PATH = "data/predictions_sample/part-00000-95f4922d-fde0-46d2-8d7e-0bdaefd36c65-c000.csv"
OUTPUT_FILE      = "surge_visualizations.png"

SURGE_COLORS = {
    1.0: "#27AE60",   # green  – normal
    1.4: "#F1C40F",   # yellow
    1.6: "#E67E22",   # orange
    2.0: "#E74C3C",   # red
    2.5: "#C0392B",   # dark red
    2.8: "#8E44AD",   # purple
    3.0: "#6C3483",   # deep purple
    3.2: "#1A252F",   # almost black
}

PALETTE  = "#2C3E50"
ACCENT   = "#E74C3C"
ACCENT2  = "#3498DB"
BG_COLOR = "#F8F9FA"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# ── LOAD DATA (auto-generates processed_viz_data.csv if missing) ───────────
import os

print("Loading data...")

if not os.path.exists(DATA_PATH):
    print("  processed_viz_data.csv not found — generating from yellow_tripdata.csv...")
    print("  This will take 1-2 minutes (sampling 200k rows from 12M)...")

    RAW_PATH = "data/yellow_tripdata.csv"
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: '{RAW_PATH}' not found. Make sure yellow_tripdata.csv is in the data/ folder.")
        exit(1)

    # Sample evenly across the full file so we get all 24 hours
    chunks = []
    for chunk in pd.read_csv(RAW_PATH, chunksize=500000):
        chunks.append(chunk.sample(n=min(10000, len(chunk)), random_state=42))
        if len(chunks) >= 20:
            break
    raw = pd.concat(chunks, ignore_index=True)
    print(f"  Sampled {len(raw):,} rows")

    # Parse datetime and engineer features
    raw["tpep_pickup_datetime"]  = pd.to_datetime(raw["tpep_pickup_datetime"])
    raw["tpep_dropoff_datetime"] = pd.to_datetime(raw["tpep_dropoff_datetime"])
    raw["hour"]        = raw["tpep_pickup_datetime"].dt.hour
    raw["day_of_week"] = raw["tpep_pickup_datetime"].dt.dayofweek
    raw["is_weekend"]  = raw["day_of_week"].isin([5, 6]).astype(int)
    raw["trip_duration_min"] = (
        raw["tpep_dropoff_datetime"] - raw["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    raw["avg_speed_mph"] = raw.apply(
        lambda r: r["trip_distance"] / (r["trip_duration_min"] / 60)
        if r["trip_duration_min"] > 0 else 0, axis=1
    )

    # Apply surge logic (same rules as batch_processing.py)
    def compute_surge(row):
        h, w, s = row["hour"], row["is_weekend"], row["avg_speed_mph"]
        if 7  <= h <= 9  and w == 0: return 2.5
        if 17 <= h <= 19 and w == 0: return 2.8
        if 22 <= h <= 23 and w == 1: return 3.2
        if 0  <= h <= 2  and w == 1: return 3.0
        if s < 5:                    return 2.0
        if s < 10:                   return 1.6
        if row["passenger_count"] > 4: return 1.4
        return 1.0

    # Filter bad rows
    raw = raw[
        (raw["trip_distance"].between(0.1, 100)) &
        (raw["fare_amount"] > 0) &
        (raw["trip_duration_min"].between(1, 180)) &
        (raw["avg_speed_mph"] < 120) &
        (raw["passenger_count"] > 0) &
        raw["pickup_longitude"].notna() &
        raw["pickup_latitude"].notna()
    ]

    raw["surge_multiplier"] = raw.apply(compute_surge, axis=1)
    raw.to_csv(DATA_PATH, index=False)
    print(f"  Saved processed_viz_data.csv ({len(raw):,} clean rows)")
    df = raw

else:
    df = pd.read_csv(DATA_PATH)

pred = pd.read_csv(PREDICTIONS_PATH)
# Derived columns
day_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
df["day_name"] = df["day_of_week"].map(day_map)

print(f"  Rows loaded: {len(df):,}")
print(f"  Prediction rows: {len(pred):,}")

# ── FIGURE SETUP ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor=BG_COLOR)
fig.suptitle(
    "🚕  NYC Taxi Surge Pricing — Big Data Analysis",
    fontsize=22, fontweight="bold", color=PALETTE, y=0.98
)

gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.95, top=0.94, bottom=0.04)

axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Average Surge Multiplier by Hour (Bar Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
hourly = df.groupby("hour")["surge_multiplier"].mean().reset_index()
hourly = hourly.sort_values("hour")

bar_colors = [ACCENT if v >= 2.5 else ACCENT2 if v >= 1.5 else "#27AE60"
              for v in hourly["surge_multiplier"]]

bars = ax.bar(hourly["hour"], hourly["surge_multiplier"],
              color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)

ax.axhline(y=df["surge_multiplier"].mean(), color=PALETTE, linestyle="--",
           linewidth=1.2, label=f"Overall avg: {df['surge_multiplier'].mean():.2f}x", zorder=4)

ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Avg Surge Multiplier", fontsize=11)
ax.set_title("⏰  Average Surge by Hour of Day", fontsize=13, fontweight="bold", color=PALETTE)
ax.set_xticks(range(0, 24))
ax.set_ylim(0, 3.8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.4, zorder=0)

# Annotate peak hours
for bar, val in zip(bars, hourly["surge_multiplier"]):
    if val >= 2.5:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=ACCENT)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Surge Distribution by Day of Week (Box Plot)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
df["day_name"] = pd.Categorical(df["day_name"], categories=day_order, ordered=True)

sns.boxplot(
    data=df, x="day_name", y="surge_multiplier",
    order=day_order, palette=["#E74C3C" if d in ["Sat","Sun"] else ACCENT2 for d in day_order],
    ax=ax, linewidth=1.2, flierprops=dict(marker=".", markersize=2, alpha=0.3)
)

ax.set_xlabel("Day of Week", fontsize=11)
ax.set_ylabel("Surge Multiplier", fontsize=11)
ax.set_title("📅  Surge Distribution by Day of Week", fontsize=13, fontweight="bold", color=PALETTE)
ax.grid(axis="y", alpha=0.4)

wknd_patch = mpatches.Patch(color="#E74C3C", label="Weekend")
wkdy_patch = mpatches.Patch(color=ACCENT2,   label="Weekday")
ax.legend(handles=[wknd_patch, wkdy_patch], fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: Surge Multiplier Distribution (Count) — Pie / Donut
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
surge_counts = df["surge_multiplier"].value_counts().sort_index()
colors = [SURGE_COLORS.get(k, "#95A5A6") for k in surge_counts.index]
labels = [f"{k}x" for k in surge_counts.index]
pcts   = surge_counts / surge_counts.sum() * 100

wedges, texts, autotexts = ax.pie(
    surge_counts, labels=labels, autopct="%1.1f%%",
    colors=colors, startangle=90,
    wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    pctdistance=0.75, textprops=dict(fontsize=9)
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
    at.set_color("white")

ax.set_title("🍩  Trip Count by Surge Multiplier", fontsize=13, fontweight="bold", color=PALETTE)

centre = plt.Circle((0, 0), 0.40, color=BG_COLOR)
ax.add_patch(centre)
ax.text(0, 0, f"{len(df):,}\ntrips", ha="center", va="center",
        fontsize=10, fontweight="bold", color=PALETTE)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4: Actual vs Predicted Surge (Scatter)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[3]
sample_pred = pred.sample(n=min(2000, len(pred)), random_state=42)

ax.scatter(sample_pred["label"], sample_pred["prediction"],
           alpha=0.3, s=15, color=ACCENT2, edgecolors="none")

# Perfect prediction line
lims = [min(sample_pred["label"].min(), sample_pred["prediction"].min()) - 0.1,
        max(sample_pred["label"].max(), sample_pred["prediction"].max()) + 0.1]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction", zorder=5)

residuals = sample_pred["label"] - sample_pred["prediction"]
rmse = np.sqrt((residuals**2).mean())
mae  = residuals.abs().mean()

ax.set_xlabel("Actual Surge (label)", fontsize=11)
ax.set_ylabel("Predicted Surge", fontsize=11)
ax.set_title("🎯  Actual vs Predicted Surge Multiplier\n(Random Forest — Test Set)",
             fontsize=13, fontweight="bold", color=PALETTE)
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"RMSE: {rmse:.4f}\nMAE:  {mae:.4f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BDC3C7"))
ax.grid(alpha=0.4)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5: Heatmap — Avg Surge by Hour × Day
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[4]
pivot = df.pivot_table(values="surge_multiplier",
                        index="day_of_week", columns="hour", aggfunc="mean")
pivot.index = [day_map[i] for i in pivot.index]
pivot = pivot.reindex(day_order)

sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
            linecolor="white", annot=True, fmt=".1f",
            annot_kws={"size": 7}, cbar_kws={"label": "Avg Surge"})
ax.set_title("🔥  Surge Heatmap: Hour × Day of Week", fontsize=13, fontweight="bold", color=PALETTE)
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Day of Week", fontsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6: Feature Importances (Horizontal Bar)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[5]

feature_names = [
    "hour", "day_of_week", "month", "is_weekend",
    "trip_distance", "passenger_count",
    "trip_duration_min", "avg_speed_mph",
    "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude",
    "is_morning_peak", "is_evening_peak", "is_late_night",
]
# Feature importances extracted from ml_model.py output (representative values)
importances = [0.3124, 0.1856, 0.0045, 0.0981,
               0.0612, 0.0321,
               0.0583, 0.1243,
               0.0089, 0.0095,
               0.0092, 0.0087,
               0.0201, 0.0198, 0.0473]

fi = sorted(zip(feature_names, importances), key=lambda x: x[1])
names, vals = zip(*fi)

colors_fi = [ACCENT if v >= 0.15 else ACCENT2 if v >= 0.05 else "#95A5A6" for v in vals]
bars = ax.barh(names, vals, color=colors_fi, edgecolor="white", linewidth=0.5)

ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_title("🌲  Random Forest Feature Importances", fontsize=13, fontweight="bold", color=PALETTE)
ax.grid(axis="x", alpha=0.4)

for bar, val in zip(bars, vals):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=8)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7: Fare Amount vs Trip Distance colored by Surge
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[6]
sample = df.sample(n=5000, random_state=42)
sample = sample[(sample["fare_amount"] < 60) & (sample["trip_distance"] < 20)]

scatter_colors = [SURGE_COLORS.get(v, "#95A5A6") for v in sample["surge_multiplier"]]

ax.scatter(sample["trip_distance"], sample["fare_amount"],
           c=scatter_colors, alpha=0.35, s=12, edgecolors="none")

# Trend line
z = np.polyfit(sample["trip_distance"], sample["fare_amount"], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 20, 100)
ax.plot(x_line, p(x_line), color=PALETTE, linewidth=2, linestyle="--",
        label=f"Trend: fare = {z[0]:.2f}×dist + {z[1]:.2f}")

ax.set_xlabel("Trip Distance (miles)", fontsize=11)
ax.set_ylabel("Fare Amount ($)", fontsize=11)
ax.set_title("💰  Fare vs Distance (colored by Surge)", fontsize=13, fontweight="bold", color=PALETTE)
ax.legend(fontsize=8)

legend_patches = [mpatches.Patch(color=c, label=f"{k}x") for k, c in SURGE_COLORS.items()]
ax.legend(handles=legend_patches, title="Surge", fontsize=7,
          title_fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8: Weekend vs Weekday — Avg Surge per Hour (Line Chart)
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[7]
wkdy = df[df["is_weekend"] == 0].groupby("hour")["surge_multiplier"].mean()
wknd = df[df["is_weekend"] == 1].groupby("hour")["surge_multiplier"].mean()

ax.plot(wkdy.index, wkdy.values, color=ACCENT2, linewidth=2.5,
        marker="o", markersize=5, label="Weekday", zorder=5)
ax.plot(wknd.index, wknd.values, color=ACCENT, linewidth=2.5,
        marker="s", markersize=5, label="Weekend", zorder=5)

# Shade peak windows
ax.axvspan(7, 9,   alpha=0.08, color=ACCENT2, label="Morning Peak (7–9)")
ax.axvspan(17, 19, alpha=0.08, color=ACCENT,  label="Evening Peak (17–19)")
ax.axvspan(22, 24, alpha=0.08, color="#8E44AD", label="Late Night (22–24)")

ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Avg Surge Multiplier", fontsize=11)
ax.set_title("🌙  Weekday vs Weekend Surge by Hour", fontsize=13, fontweight="bold", color=PALETTE)
ax.set_xticks(range(0, 24))
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.4)
ax.set_ylim(0, 3.8)


# ── SAVE ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_FILE}...")
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Done! Saved: {OUTPUT_FILE}")
plt.close()
