import os
import json
import pytz
from datetime import datetime, timedelta
from minio import Minio
from minio.error import S3Error
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

class MinioService:
    def __init__(self):
        endpoint    = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key  = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key  = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        bucket_name = os.getenv("MINIO_BUCKET", "warehouse")
        secure      = os.getenv("MINIO_SECURE", "False").lower() in ("true", "1")

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name

        # Ensure the bucket exists
        try:
            if not self.client.bucket_exists(self.bucket_name):
                logging.info(f"Bucket '{self.bucket_name}' not found. Creating it...")
                self.client.make_bucket(self.bucket_name)
                logging.info(f"Bucket '{self.bucket_name}' created.")
        except S3Error as err:
            logging.error(f"Error checking/creating bucket '{self.bucket_name}': {err}")
            raise

    def list_all_objects(self, limit=None):
        try:
            all_objects = list(self.client.list_objects(self.bucket_name, recursive=True))
            current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)

            thirty_days_ago = current_time - timedelta(days=30)
            ten_days_ago = current_time - timedelta(days=20)

            # Filter objects modified between 30 and 10 days ago
            latest_objects = [
                obj for obj in all_objects
                if thirty_days_ago < obj.last_modified <= ten_days_ago
            ]
            logging.info(f"Found {len(latest_objects)} objects in bucket '{self.bucket_name}'.")
        except S3Error as err:
            logging.error(f"Error listing objects in '{self.bucket_name}': {err}")
            return []



        # newest-first
        sorted_objects = sorted(
            latest_objects,
            key=lambda o: o.last_modified,
            reverse=True
        )

        return sorted_objects[:limit] if limit and limit > 0 else sorted_objects

    def get_object_data(self, object_name):
        """Returns the raw string data from the object."""
        try:
            resp = self.client.get_object(self.bucket_name, object_name)
            raw = resp.read().decode('utf-8')
            return raw
        except S3Error as err:
            logging.error(f"Error retrieving '{object_name}': {err}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing '{object_name}': {e}")
            return None
        finally:
            try:
                resp.close()
                resp.release_conn()
            except:
                pass
 