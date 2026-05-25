"""
Spark MLlib — Surge Pricing Prediction Model (2015/2016 Dataset)
Uses pickup_longitude/latitude instead of PULocationID
"""
 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when, round as _round
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
 
 
def main():
    spark = (
        SparkSession.builder
        .appName("SurgePricing_MLPipeline")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print("  Spark session ready — loading features...")
 
    # ── Load features saved by batch_processing.py ────────────────────────────
    df = spark.read.parquet("/home/jovyan/data/taxi_features")
    print(f"📂  Feature rows: {df.count():,}")
    df.printSchema()
 
    # ── Additional derived features ───────────────────────────────────────────
    df = (
        df
        .withColumn("is_morning_peak", when(col("hour").between(7, 9),   1).otherwise(0))
        .withColumn("is_evening_peak", when(col("hour").between(17, 20), 1).otherwise(0))
        .withColumn("is_late_night",
            when(col("hour").between(22, 23), 1)
            .when(col("hour").between(0, 2),  1)
            .otherwise(0)
        )
        .filter(col("surge_multiplier").isNotNull())
        .withColumnRenamed("surge_multiplier", "label")
    )
 
    # ── Assemble features ─────────────────────────────────────────────────────
    assembler = VectorAssembler(
        inputCols=[
            "hour", "day_of_week", "month", "is_weekend",
            "trip_distance", "passenger_count",
            "trip_duration_min", "avg_speed_mph",
            "pickup_longitude", "pickup_latitude",
            "dropoff_longitude", "dropoff_latitude",
            "is_morning_peak", "is_evening_peak", "is_late_night",
        ],
        outputCol="features",
        handleInvalid="skip"
    )
 
    # ── Random Forest Model ───────────────────────────────────────────────────
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        seed=42
    )
 
    pipeline = Pipeline(stages=[assembler, rf])
 
    # ── Train / Test Split ────────────────────────────────────────────────────
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train.count():,} | Test: {test.count():,}")
 
    print("\n  Training Random Forest...")
    model = pipeline.fit(train)
    predictions = model.transform(test)
 
    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
 
    rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    r2   = evaluator.setMetricName("r2").evaluate(predictions)
    mae  = evaluator.setMetricName("mae").evaluate(predictions)
 
    print("\n" + "="*45)
    print("  MODEL EVALUATION RESULTS")
    print("="*45)
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print("="*45)
 
    # ── Feature Importances ───────────────────────────────────────────────────
    feature_names = [
        "hour", "day_of_week", "month", "is_weekend",
        "trip_distance", "passenger_count",
        "trip_duration_min", "avg_speed_mph",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "is_morning_peak", "is_evening_peak", "is_late_night",
    ]
    importances = model.stages[-1].featureImportances.toArray()
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
 
    print("\n  Feature Importances:")
    for name, imp in ranked[:8]:
        bar = "█" * int(imp * 50)
        print(f"   {name:<22} {imp:.4f}  {bar}")
 
    # ── Save Model ────────────────────────────────────────────────────────────
    model.save("/home/jovyan/data/surge_model")
    print("\n  Model saved to /home/jovyan/data/surge_model")
 
    # ── Save predictions sample ───────────────────────────────────────────────
    (
        predictions
        .select("label", "prediction", "hour", "day_of_week", "is_weekend")
        .limit(5000)
        .write
        .mode("overwrite")
        .csv("/home/jovyan/data/predictions_sample", header=True)
    )
    print("  Predictions sample saved.")
    spark.stop()
 
 
if __name__ == "__main__":
    main()