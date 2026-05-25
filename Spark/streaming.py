from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("StreamingApp") \
    .config("spark.sql.shuffle.partitions", "2") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("timestamp",         StringType()),
    StructField("zone",              StringType()),
    StructField("ride_requests",     IntegerType()),
    StructField("available_drivers", IntegerType()),
    StructField("hour",              IntegerType()),
    StructField("is_weekend",        BooleanType()),
    StructField("weather_code",      IntegerType()),
])

raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "ride_requests") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

df = raw.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

df = df.withColumn("event_time", to_timestamp("timestamp"))

windowed = df \
    .withWatermark("event_time", "2 minutes") \
    .groupBy(window("event_time", "5 minutes"), "zone") \
    .agg(
        sum("ride_requests").alias("total_requests"),
        avg("available_drivers").alias("avg_drivers"),
        first("weather_code").alias("weather_code"),
        first("is_weekend").alias("is_weekend")
    ) \
    .withColumn("Demand_Supply_Ratio",
        round(col("total_requests") / col("avg_drivers"), 2)
    ) \
    .withColumn("surge_multiplier",
        when(col("Demand_Supply_Ratio") >= 3.5, 3.5)
        .when(col("Demand_Supply_Ratio") >= 2.5, 2.8)
        .when(col("Demand_Supply_Ratio") >= 1.8, 2.2)
        .when(col("Demand_Supply_Ratio") >= 1.3, 1.8)
        .when(col("Demand_Supply_Ratio") >= 1.0, 1.4)
        .otherwise(1.0)
    )

query = windowed.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="15 seconds") \
    .start()

print(" Streaming started! Waiting for data...")
query.awaitTermination()