import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, date_format, to_json
from pyspark.sql.types import StructType, StructField, StringType, LongType

# Snowflake connection options
SF_OPTS = {
    "sfURL": "ilgybpn-wg79816.snowflakecomputing.com",
    "sfUser": "OMAR",
    "sfPassword": os.environ.get("SNOWFLAKE_PWD"),
    "sfDatabase": "PROM_DB",
    "sfSchema": "MY_SCHEMA",
    "sfWarehouse": "COMPUTE_WH",
    "sfRole": "ACCOUNTADMIN"
}

PROM_SCHEMA = StructType([
    StructField("labels", StructType([
        StructField("__name__", StringType(), True),
        StructField("cpu", StringType(), True),
        StructField("instance", StringType(), True),
        StructField("job", StringType(), True),
        StructField("mode", StringType(), True),
        StructField("os_type", StringType(), True),
        StructField("vm", StringType(), True),
        StructField("name", StringType(), True),
        StructField("state", StringType(), True),
        StructField("type", StringType(), True)
    ]), True),
    StructField("name", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("value", StringType(), True)
])

def process_kafka_stream(spark, bootstrap_servers, topic_name):
    """
    Create a streaming DataFrame from Kafka with proper configurations
    """
    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", bootstrap_servers)
           .option("subscribe", topic_name)
           .option("startingOffsets", "earliest")
           .option("kafka.request.timeout.ms", "120000")
           .option("kafka.session.timeout.ms", "120000")
           .option("failOnDataLoss", "false")
           .option("kafka.max.poll.records", "100")
           .load())

    # Parse the JSON and extract relevant fields
    return (raw
            .selectExpr(
                "CAST(offset AS LONG) AS kafka_offset",
                "timestamp AS kafka_ingest_ts",
                "CAST(value AS STRING) AS json_str"
            )
            .select(
                col("kafka_offset"),
                col("kafka_ingest_ts"),
                from_json(col("json_str"), PROM_SCHEMA).alias("j")
            )
            .select(
                col("kafka_offset"),
                col("kafka_ingest_ts"),
                col("j.name").alias("metric_name"),
                to_timestamp(col("j.timestamp")).alias("event_ts"),
                col("j.value").alias("metric_value"),
                to_json(col("j.labels")).alias("labels")  # Store labels as JSON string
            ))

def write_to_staging(batch_df, batch_id):
    """
    Write each batch to Snowflake STG_METRICS table
    """
    print(f"=== Batch {batch_id} arrived ===")

    try:
        if batch_df.isEmpty():
            print(f"Batch {batch_id} is empty, skipping.")
            return

        batch_df.cache()

        # Format timestamps as strings
        formatted_df = batch_df.withColumn(
            "kafka_ingest_ts",
            date_format(col("kafka_ingest_ts"), "yyyy-MM-dd HH:mm:ss")
        ).withColumn(
            "event_ts",
            date_format(col("event_ts"), "yyyy-MM-dd HH:mm:ss")
        )

        # Select columns to match STG_METRICS schema
        final_df = formatted_df.select(
            "kafka_offset",
            "kafka_ingest_ts",
            "metric_name",
            "event_ts",
            "metric_value",
            "labels"
        )

        print("Data sample:")
        final_df.show(5, truncate=False)
        cnt = final_df.count()
        print(f"Total rows to write: {cnt}")

        if cnt == 0:
            print("No data to write, skipping.")
            return

        # Write to Snowflake
        final_df.write \
            .format("snowflake") \
            .options(**SF_OPTS) \
            .option("dbtable", "STG_METRICS") \
            .mode("append") \
            .save()

        print(f"Successfully wrote {cnt} records to STG_METRICS")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error writing batch {batch_id}: {e}")

    finally:
        batch_df.unpersist()

if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder \
        .appName("KafkaToSnowflakeStaging") \
        .config("spark.streaming.kafka.consumer.poll.ms", "512") \
        .config("spark.streaming.kafka.allowNonConsecutiveOffsets", "true") \
        .getOrCreate()

    if not os.environ.get("SNOWFLAKE_PWD"):
        raise ValueError("Please set SNOWFLAKE_PWD environment variable")

    # Start the streaming job
    kafka_host = "localhost:29092"
    topic = "prometheus_metrics"

    try:
        print(f"Starting Kafka ingestion from {kafka_host} topic {topic}")
        stream = process_kafka_stream(spark, kafka_host, topic)

        query = (stream.writeStream
                .outputMode("append")
                .foreachBatch(write_to_staging)
                .option("checkpointLocation", "./checkpoints/staging_ingest")
                .trigger(processingTime="10 seconds")
                .start())

        query.awaitTermination()
    except Exception as e:
        print(f"Fatal error in streaming job: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down Spark session")
        spark.stop()