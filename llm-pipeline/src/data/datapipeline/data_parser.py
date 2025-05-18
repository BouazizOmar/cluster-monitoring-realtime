import json
import logging
from datetime import datetime
import pytz

class DataParser:
    @staticmethod
    def parse_minio_data(data):
        """
        Parse MinIO JSON string logs into structured format.
        Each line is expected to be a JSON string.
        """
        records = []
        for line in data.strip().split("\n"):
            try:
                # Remove extra quotes if present and convert to JSON.
                record = json.loads(line.strip("'"))
                # Convert the timestamp string to a datetime (assumed UTC)
                record["timestamp"] = datetime.strptime(
                    record["timestamp"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=pytz.UTC)
                records.append(record)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid line: {line} due to error: {e}")
        return records
