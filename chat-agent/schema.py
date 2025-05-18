"""
Schema metadata definition for the galaxy schema
"""

def get_schema_metadata():
    """Return the hardcoded metadata about the galaxy schema structure."""
    metadata = {
        "fact_tables": [
            {
                "name": "FACT_METRIC_VALUES",
                "columns": [
                    {"name": "METRIC_VALUE_ID", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "METRIC_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_METRIC"},
                    {"name": "HOST_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_HOST"},
                    {"name": "MODE_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_MODE"},
                    {"name": "CPU_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_CPU"},
                    {"name": "VM_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_VM"},
                    {"name": "TIME_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_TIME"},
                    {"name": "VALUE", "type": "FLOAT", "description": "Metric measurement value"},
                    {"name": "KAFKA_OFFSET", "type": "NUMBER(38,0)", "description": "Kafka offset for data tracking"},
                    {"name": "KAFKA_TIMESTAMP", "type": "TIMESTAMP_NTZ(9)", "description": "Timestamp from Kafka"},
                    {"name": "INGEST_TIMESTAMP", "type": "TIMESTAMP_NTZ(9)", "description": "Timestamp of data ingestion"}
                ]
            },
            {
                "name": "FACT_PROCESS_SERVICE_VALUES",
                "columns": [
                    {"name": "PROCESS_SERVICE_VALUE_ID", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "METRIC_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_METRIC"},
                    {"name": "HOST_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_HOST"},
                    {"name": "MODE_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_MODE"},
                    {"name": "CPU_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_CPU"},
                    {"name": "VM_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_VM"},
                    {"name": "TIME_KEY", "type": "NUMBER(38,0)", "description": "Foreign key to DIM_TIME"},
                    {"name": "SERVICE_NAME", "type": "VARCHAR(16777216)", "description": "Name of the service"},
                    {"name": "STATE", "type": "VARCHAR(16777216)", "description": "Current state of the service"},
                    {"name": "TYPE", "type": "VARCHAR(16777216)", "description": "Type of service or process"},
                    {"name": "VALUE", "type": "VARCHAR(16777216)", "description": "Service metric value"},
                    {"name": "KAFKA_OFFSET", "type": "NUMBER(38,0)", "description": "Kafka offset for data tracking"},
                    {"name": "KAFKA_TIMESTAMP", "type": "TIMESTAMP_NTZ(9)", "description": "Timestamp from Kafka"},
                    {"name": "INGEST_TIMESTAMP", "type": "TIMESTAMP_NTZ(9)", "description": "Timestamp of data ingestion"}
                ]
            }
        ],
        "dimension_tables": [
            {
                "name": "DIM_CPU",
                "columns": [
                    {"name": "CPU_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "CPU_ID", "type": "VARCHAR(16777216)", "description": "CPU identifier"}
                ]
            },
            {
                "name": "DIM_HOST",
                "columns": [
                    {"name": "HOST_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "INSTANCE", "type": "VARCHAR(16777216)", "description": "Instance identifier"},
                    {"name": "JOB", "type": "VARCHAR(16777216)", "description": "Job name or identifier"}
                ]
            },
            {
                "name": "DIM_METRIC",
                "columns": [
                    {"name": "METRIC_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "METRIC_NAME", "type": "VARCHAR(16777216)", "description": "Name of the metric"}
                ]
            },
            {
                "name": "DIM_MODE",
                "columns": [
                    {"name": "MODE_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "MODE", "type": "VARCHAR(16777216)", "description": "Operation mode"}
                ]
            },
            {
                "name": "DIM_TIME",
                "columns": [
                    {"name": "TIME_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "TS", "type": "TIMESTAMP_NTZ(9)", "description": "Full timestamp"},
                    {"name": "DAY", "type": "DATE", "description": "Date part"},
                    {"name": "HOUR", "type": "NUMBER(38,0)", "description": "Hour of day"},
                    {"name": "MINUTE", "type": "NUMBER(38,0)", "description": "Minute of hour"}
                ]
            },
            {
                "name": "DIM_VM",
                "columns": [
                    {"name": "VM_KEY", "type": "NUMBER(38,0)", "description": "Primary key"},
                    {"name": "VM", "type": "VARCHAR(16777216)", "description": "Virtual machine identifier"},
                    {"name": "OS_TYPE", "type": "VARCHAR(16777216)", "description": "Operating system type"}
                ]
            }
        ],
        "relationships": [
            {"parent_table": "DIM_METRIC", "parent_column": "METRIC_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "METRIC_KEY"},
            {"parent_table": "DIM_HOST", "parent_column": "HOST_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "HOST_KEY"},
            {"parent_table": "DIM_MODE", "parent_column": "MODE_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "MODE_KEY"},
            {"parent_table": "DIM_CPU", "parent_column": "CPU_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "CPU_KEY"},
            {"parent_table": "DIM_VM", "parent_column": "VM_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "VM_KEY"},
            {"parent_table": "DIM_TIME", "parent_column": "TIME_KEY", "foreign_table": "FACT_METRIC_VALUES", "foreign_column": "TIME_KEY"},
            {"parent_table": "DIM_METRIC", "parent_column": "METRIC_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "METRIC_KEY"},
            {"parent_table": "DIM_HOST", "parent_column": "HOST_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "HOST_KEY"},
            {"parent_table": "DIM_MODE", "parent_column": "MODE_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "MODE_KEY"},
            {"parent_table": "DIM_CPU", "parent_column": "CPU_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "CPU_KEY"},
            {"parent_table": "DIM_VM", "parent_column": "VM_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "VM_KEY"},
            {"parent_table": "DIM_TIME", "parent_column": "TIME_KEY", "foreign_table": "FACT_PROCESS_SERVICE_VALUES", "foreign_column": "TIME_KEY"}
        ],
        "business_context": {
            "overall": "This is a monitoring and metrics system with a galaxy schema. It tracks various metrics across hosts, CPUs, VMs, and services over time.",
            "fact_metric_values": "Contains numeric metric measurements across different dimensions (hosts, CPUs, VMs) over time",
            "fact_process_service_values": "Tracks service and process states and metrics across different dimensions",
            "metrics": "Various system and service metrics being monitored",
            "time_granularity": "Metrics can be analyzed at minute, hour, and day levels"
        }
    }
    return metadata