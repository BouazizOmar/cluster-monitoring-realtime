#!/bin/bash

# Set Snowflake password
export SNOWFLAKE_PWD=your_snowflake_password_here

# Run the Spark job
$SPARK_HOME/bin/spark-submit \
--master "local[*]" \
--driver-memory 4g \
--executor-memory 4g \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--packages \
org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,\
org.apache.kafka:kafka-clients:3.4.0,\
org.apache.commons:commons-pool2:2.11.1,\
net.snowflake:spark-snowflake_2.12:3.1.1,\
net.snowflake:snowflake-jdbc:3.19.0 \
staging_ingestion.py  