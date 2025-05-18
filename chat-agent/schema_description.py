# schema_description.py
# Schema description for the monitoring database structure

SCHEMA_DESCRIPTION = """
The Snowflake database contains the following tables:

1. FACT_METRIC_VALUES
   - METRIC_VALUE_ID (INTEGER): Unique identifier for metric value
   - METRIC_KEY (INTEGER): Foreign key to DIM_METRIC
   - HOST_KEY (INTEGER): Foreign key to DIM_HOST
   - MODE_KEY (INTEGER): Foreign key to DIM_MODE
   - CPU_KEY (INTEGER): Foreign key to DIM_CPU
   - VM_KEY (INTEGER): Foreign key to DIM_VM
   - TIME_KEY (INTEGER): Foreign key to DIM_TIME
   - VALUE (FLOAT): Actual metric value
   - KAFKA_OFFSET (INTEGER): Kafka message offset
   - KAFKA_TIMESTAMP (TIMESTAMP_NTZ): Timestamp from Kafka
   - INGEST_TIMESTAMP (TIMESTAMP_NTZ): Timestamp of data ingestion

2. FACT_PROCESS_SERVICE_VALUES
   - PROCESS_SERVICE_VALUE_ID (INTEGER): Unique identifier for service value
   - METRIC_KEY (INTEGER): Foreign key to DIM_METRIC
   - HOST_KEY (INTEGER): Foreign key to DIM_HOST
   - MODE_KEY (INTEGER): Foreign key to DIM_MODE
   - CPU_KEY (INTEGER): Foreign key to DIM_CPU
   - VM_KEY (INTEGER): Foreign key to DIM_VM
   - TIME_KEY (INTEGER): Foreign key to DIM_TIME
   - SERVICE_NAME (STRING): Name of the service
   - STATE (STRING): Service state (active, failed, inactive, etc.)
   - TYPE (STRING): Service type
   - VALUE (FLOAT): Metric value
   - KAFKA_OFFSET (INTEGER): Kafka message offset
   - KAFKA_TIMESTAMP (TIMESTAMP_NTZ): Timestamp from Kafka
   - INGEST_TIMESTAMP (TIMESTAMP_NTZ): Timestamp of data ingestion

3. DIM_VM
   - VM_KEY (INTEGER): Primary key
   - VM (STRING): VM name (Ubuntu, Lubuntu, Lubuntu V2)
   - OS_TYPE (STRING): Operating system type (linux)

4. DIM_METRIC
   - METRIC_KEY (INTEGER): Primary key
   - METRIC_NAME (STRING): Name of the metric (e.g., node_memory_MemTotal_bytes)

5. DIM_HOST
   - HOST_KEY (INTEGER): Primary key
   - INSTANCE (STRING): Instance identifier (IP:port)
   - JOB (STRING): Job identifier

6. DIM_CPU
   - CPU_KEY (INTEGER): Primary key
   - CPU_ID (INTEGER): CPU identifier

7. DIM_MODE
   - MODE_KEY (INTEGER): Primary key
   - MODE (STRING): Mode name (nice, user, system, etc.)

8. DIM_TIME
   - TIME_KEY (INTEGER): Primary key
   - TS (TIMESTAMP_NTZ): Timestamp
   - DAY (DATE): Day
   - HOUR (INTEGER): Hour
   - MINUTE (INTEGER): Minute

Important Memory Metrics:
- node_memory_MemTotal_bytes (METRIC_KEY = 41): Total memory in bytes
- node_memory_MemAvailable_bytes (METRIC_KEY = 78): Available memory in bytes
- node_memory_MemFree_bytes (METRIC_KEY = 66): Free memory in bytes
- node_memory_Cached_bytes (METRIC_KEY = 65): Cached memory in bytes
- node_memory_Active_bytes (METRIC_KEY = 26): Active memory in bytes

Important CPU Metrics:
- node_cpu_seconds_total (METRIC_KEY = 59): CPU seconds total
- node_cpu_guest_seconds_total (METRIC_KEY = 1): CPU guest seconds total
- node_cpu_online (METRIC_KEY = 72): CPU online status

Memory Usage Calculation:
Memory usage percentage is calculated as: ((Total - Available) / Total) * 100
This requires joining with DIM_METRIC to get both node_memory_MemTotal_bytes and node_memory_MemAvailable_bytes metrics.
""" 