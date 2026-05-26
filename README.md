# 🚕 RideWave — Real-Time Surge Pricing Intelligence Platform

> **Big Data Analytics — Final Year Project**
> University of Central Punjab · 2024–2025

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alisufyaan-surgepricingproject-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.0-orange.svg)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.5.0-black.svg)](https://kafka.apache.org/)

---

## 📌 Overview

RideWave is a real-time surge pricing intelligence platform built on a modern Big Data pipeline. The system processes **12.2 million NYC Yellow Taxi trip records** to predict dynamic surge pricing multipliers using Apache Kafka, PySpark, Spark MLlib, and an interactive Streamlit dashboard.

The platform mirrors how real-world ride-hailing services like Uber and Lyft adjust fares in real time based on demand, traffic, and time-of-day patterns.

---

## 🔴 Live Dashboard

**👉 [Open RideWave Dashboard](https://alisufyaan-surgepricingproject-app.streamlit.app)**

> If the app shows "sleeping", click **Wake up** — it starts in ~30 seconds.

---

## 🏗️ System Architecture

```
NYC Taxi Data (12.2M rows)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Apache Kafka   │────▶│  PySpark Batch   │
│  (Producer →    │     │  Processing      │
│   Consumer)     │     │  + Feature Eng.  │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
           ┌──────────────┐ ┌────────┐ ┌──────────────┐
           │ Spark Stream │ │ MLlib  │ │   Parquet    │
           │ (5-min window│ │Random  │ │   Storage    │
           │  windowing)  │ │Forest  │ └──────────────┘
           └──────────────┘ └───┬────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Streamlit Dashboard  │
                    │  Matplotlib · Seaborn │
                    │  8 Charts · Filters   │
                    └───────────────────────┘
```

---

## ⚡ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Streaming | Apache Kafka | 3.5.0 |
| Processing | Apache Spark / PySpark | 3.5.0 |
| Machine Learning | Spark MLlib (Random Forest) | 3.5.0 |
| Containerization | Docker / Docker Compose | Latest |
| Dashboard | Streamlit | 1.35.0 |
| Visualization | Matplotlib / Seaborn | 3.9 / 0.13 |
| Data Handling | Pandas / NumPy | 2.2 / 1.26 |
| Language | Python | 3.10+ |

---

## 📊 Dataset

- **Source:** NYC Yellow Taxi Trip Data — March 2016 (NYC TLC)
- **Raw Records:** 12,210,952 trips
- **Clean Sample Used:** ~197,000 records
- **Key Fields:** pickup/dropoff datetime, trip distance, fare amount, passenger count, GPS coordinates

### Engineered Features

| Feature | Description |
|---------|-------------|
| `hour` | Hour of pickup (0–23) — strongest predictor |
| `day_of_week` | Day number (0=Monday to 6=Sunday) |
| `is_weekend` | Binary flag — Saturday or Sunday |
| `trip_duration_min` | Duration in minutes |
| `avg_speed_mph` | Distance ÷ duration — traffic congestion proxy |
| `is_morning_peak` | Hours 7–9 on weekdays |
| `is_evening_peak` | Hours 17–20 on weekdays |
| `is_late_night` | Hours 22–23 and 0–2 |

---

## 🔥 Surge Pricing Logic

| Condition | Multiplier | Reason |
|-----------|-----------|--------|
| Hour 7–9 · Weekday | **2.5×** | Morning rush hour |
| Hour 17–19 · Weekday | **2.8×** | Evening rush — peak demand |
| Hour 22–23 · Weekend | **3.2×** | Weekend nightlife peak |
| Hour 0–2 · Weekend | **3.0×** | Post-midnight weekend demand |
| Avg Speed < 5 mph | **2.0×** | Heavy traffic congestion |
| Avg Speed < 10 mph | **1.6×** | Slow/moderate traffic |
| Passenger Count > 4 | **1.4×** | High occupancy |
| All other | **1.0×** | Normal fare |

---

## 🤖 ML Model Results

| Metric | Score |
|--------|-------|
| Algorithm | Random Forest Regressor |
| Number of Trees | 100 |
| Max Depth | 8 |
| Train/Test Split | 80% / 20% |
| **RMSE** | **0.0663** |
| **MAE** | **0.0418** |
| **R²** | **> 0.97** |

**Top Feature Importances:**
1. `hour` — 31.2%
2. `day_of_week` — 18.6%
3. `avg_speed_mph` — 12.4%
4. `is_weekend` — 9.8%
5. `trip_distance` — 6.1%

---

## 📈 Dashboard Features

- **Sidebar Filters** — Hour range, day of week, surge level, weekday/weekend toggle
- **KPI Cards** — Avg Surge · Peak Surge · Total Trips · Avg Fare · Model RMSE
- **8 Visualizations** — Hourly bar, day-of-week boxplot, surge donut, actual vs predicted scatter, heatmap, feature importances, fare vs distance, weekday vs weekend line chart
- **Raw Data Table** — Filtered dataset preview
- **Predictions Table** — Model output sample

---

## 📁 Project Structure

```
SurgeProject/
├── app.py                          # Streamlit dashboard
├── visualizations.py               # Matplotlib + Seaborn charts (8 charts)
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Kafka + Zookeeper setup
├── Kafka/
│   ├── producer.py                 # Ride-request event producer
│   └── surge_output_consumer.py    # Surge prediction consumer
├── Spark/
│   ├── batch_processing.py         # Feature engineering + surge labeling
│   ├── ml_model.py                 # Random Forest training + evaluation
│   └── streaming.py                # Spark Structured Streaming
└── data/
    ├── yellow_tripdata.csv          # Raw NYC taxi data (not in repo — too large)
    ├── processed_viz_data.csv       # Processed sample (in repo)
    └── predictions_sample/          # Model predictions CSV
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Docker Desktop (for Kafka)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AliSufyaan/SurgePricingProject.git
cd SurgePricingProject
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Visualizations
```bash
python visualizations.py
```
> First run takes 1–2 minutes to process data. Subsequent runs are instant.

### 4. Run the Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### 5. Start Kafka Pipeline (Optional)
```bash
docker-compose up
```

---

## 👥 Team

| Name | Role | LinkedIn |
|------|------|---------|
| **Ahmad Munir Sheikh** | Data Engineer — Kafka, PySpark Batch, Docker | [Profile](https://www.linkedin.com/in/ahmad-munir-sheikh-686b7a365/) |
| **Ali Akbar** | ML Engineer — Random Forest, Spark Streaming | [Profile](https://www.linkedin.com/in/ali-akbar-39117b314/) |
| **Ali Sufyyan** | Frontend & Visualization — Dashboard, Charts, Deployment | [Profile](https://www.linkedin.com/in/ali-sufyyan) |

---

## 📄 License

This project was developed for academic purposes at the University of Central Punjab.

---

<p align="center">Built with ❤️ using Kafka · PySpark · Spark MLlib · Streamlit</p>
