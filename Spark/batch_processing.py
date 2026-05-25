"""
Spark Batch Processing — NYC Taxi Data Analysis (2015/2016 Dataset)
"""
 
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, month,
    when, avg, count, sum as _sum,
    round as _round,
    to_timestamp, unix_timestamp, lit,
    stddev, max as _max, min as _min,
)
 
 
def main():
    spark = (
        SparkSession.builder
        .appName("SurgePricing_BatchProcessing")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("  Spark session ready — loading NYC Taxi data...")
 
    # ── 1. Load raw CSV ───────────────────────────────────────────────────────
    raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("/home/jovyan/data/yellow_tripdata.csv")
    )
 
    print(f"  Raw rows: {raw.count():,}")
    raw.printSchema()
 
    # ── 2. Feature Engineering ────────────────────────────────────────────────
    df = (
        raw
        .withColumn("pickup_dt",  to_timestamp(col("tpep_pickup_datetime")))
        .withColumn("dropoff_dt", to_timestamp(col("tpep_dropoff_datetime")))
        # Time features
        .withColumn("hour",        hour("pickup_dt"))
        .withColumn("day_of_week", dayofweek("pickup_dt"))
        .withColumn("month",       month("pickup_dt"))
        .withColumn("is_weekend",  (dayofweek("pickup_dt").isin(1, 7)).cast("int"))
        # Trip duration in minutes
        .withColumn(
            "trip_duration_min",
            _round((unix_timestamp("dropoff_dt") - unix_timestamp("pickup_dt")) / 60, 2)
        )
        # Speed proxy
        .withColumn(
            "avg_speed_mph",
            when(col("trip_duration_min") > 0,
                 _round(col("trip_distance") / (col("trip_duration_min") / 60), 2)
            ).otherwise(lit(0))
        )
        # Surge multiplier label (rule-based)
        .withColumn(
            "surge_multiplier",
            when((col("hour").between(7, 9))    & (col("is_weekend") == 0), 2.5)
            .when((col("hour").between(17, 19)) & (col("is_weekend") == 0), 2.8)
            .when((col("hour").between(22, 23)) & (col("is_weekend") == 1), 3.2)
            .when((col("hour").between(0, 2))   & (col("is_weekend") == 1), 3.0)
            .when(col("avg_speed_mph") < 5,    2.0)
            .when(col("avg_speed_mph") < 10,   1.6)
            .when(col("passenger_count") > 4,  1.4)
            .otherwise(1.0)
        )
        # Clean filter
        .filter(col("trip_distance").between(0.1, 100))
        .filter(col("fare_amount") > 0)
        .filter(col("trip_duration_min").between(1, 180))
        .filter(col("avg_speed_mph") < 120)
        .filter(col("passenger_count") > 0)
        .filter(col("pickup_longitude").isNotNull())
        .filter(col("pickup_latitude").isNotNull())
    )
 
    print(f"  Clean rows after filtering: {df.count():,}")
 
    # ── 3. Spark SQL Analytics ────────────────────────────────────────────────
    df.createOrReplaceTempView("taxi_trips")
 
    print("\n── Query 1: Average surge by hour ──")
    spark.sql("""
        SELECT
            hour,
            ROUND(AVG(surge_multiplier), 3)  AS avg_surge,
            ROUND(AVG(fare_amount), 2)        AS avg_fare,
            ROUND(AVG(trip_distance), 2)      AS avg_distance,
            COUNT(*)                          AS trip_count
        FROM taxi_trips
        GROUP BY hour
        ORDER BY avg_surge DESC
        LIMIT 10
    """).show()
 
    print("── Query 2: Top pickup areas by demand ──")
    spark.sql("""
        SELECT
            ROUND(pickup_longitude, 2)       AS area_lon,
            ROUND(pickup_latitude, 2)        AS area_lat,
            COUNT(*)                         AS total_trips,
            ROUND(AVG(surge_multiplier), 3)  AS avg_surge,
            ROUND(AVG(fare_amount), 2)       AS avg_fare
        FROM taxi_trips
        GROUP BY ROUND(pickup_longitude, 2), ROUND(pickup_latitude, 2)
        ORDER BY total_trips DESC
        LIMIT 15
    """).show()
 
    print("── Query 3: Weekend vs Weekday ──")
    spark.sql("""
        SELECT
            CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            ROUND(AVG(surge_multiplier), 3)  AS avg_surge,
            ROUND(AVG(fare_amount), 2)       AS avg_fare,
            ROUND(AVG(trip_distance), 2)     AS avg_distance,
            ROUND(AVG(trip_duration_min), 1) AS avg_duration_min,
            COUNT(*)                         AS total_trips
        FROM taxi_trips
        GROUP BY is_weekend
        ORDER BY avg_surge DESC
    """).show()
 
    print("── Query 4: Surge distribution percentiles ──")
    spark.sql("""
        SELECT
            PERCENTILE_APPROX(surge_multiplier, 0.25) AS p25,
            PERCENTILE_APPROX(surge_multiplier, 0.50) AS median,
            PERCENTILE_APPROX(surge_multiplier, 0.75) AS p75,
            PERCENTILE_APPROX(surge_multiplier, 0.90) AS p90,
            PERCENTILE_APPROX(surge_multiplier, 0.99) AS p99,
            ROUND(AVG(surge_multiplier), 3)            AS mean,
            ROUND(STDDEV(surge_multiplier), 3)         AS std_dev,
            MAX(surge_multiplier)                      AS max_surge
        FROM taxi_trips
    """).show()
 
    print("── Query 5: Peak hour demand windows ──")
    spark.sql("""
        SELECT
            hour,
            day_of_week,
            COUNT(*)                        AS trip_count,
            ROUND(AVG(surge_multiplier), 2) AS avg_surge,
            ROUND(MAX(surge_multiplier), 2) AS max_surge,
            ROUND(AVG(avg_speed_mph), 1)    AS avg_speed_mph
        FROM taxi_trips
        WHERE hour IN (7, 8, 9, 17, 18, 19, 22, 23, 0, 1)
        GROUP BY hour, day_of_week
        ORDER BY avg_surge DESC
        LIMIT 20
    """).show()
 
    # ── 4. Save processed features for ML ─────────────────────────────────────
    print("\n  Saving cleaned features for ML training...")
    (
        df.select(
            "hour", "day_of_week", "month", "is_weekend",
            "trip_distance", "passenger_count",
            "trip_duration_min", "avg_speed_mph",
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
            "fare_amount", "surge_multiplier"
        )
        .write
        .mode("overwrite")
        .parquet("/home/jovyan/data/taxi_features")
    )
 
    print("  Batch processing complete. Features saved to /home/jovyan/data/taxi_features")
    spark.stop()
 
 
if __name__ == "__main__":
    main()
 