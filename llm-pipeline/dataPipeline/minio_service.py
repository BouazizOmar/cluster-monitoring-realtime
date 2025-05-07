import os
import pytz
from datetime import datetime
from minio import Minio
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

class MinioService:
    def __init__(self):
        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "")
        secret_key = os.getenv("MINIO_SECRET_KEY", "")
        bucket_name = os.getenv("MINIO_BUCKET_NAME", "warehouse")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name

    def list_all_objects(self, limit=None):
        all_objects = list(self.client.list_objects(self.bucket_name, recursive=True))
        sorted_objects = sorted(all_objects, key=lambda obj: obj.last_modified, reverse=True)
        return sorted_objects[:limit] if limit and limit > 0 else sorted_objects

    def get_object_data(self, object_name):
        response = None
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return response.read().decode('utf-8')
        except Exception as e:
            print(f"Error retrieving object {object_name}: {e}")
            return None
        finally:
            if response:
                response.close()
                response.release_conn()
