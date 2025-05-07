import os
from datetime import datetime, timedelta, timezone
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import logging
import pandas as pd

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), "dataPipeline"))

# Import modules
from dataPipeline.minio_service import MinioService
from dataPipeline.data_parser import DataParser
from dataPipeline.metrics_processor import MetricsProcessor
from dataPipeline.snapshot_generator import SnapshotGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("prompt_api.log"), logging.StreamHandler()]
)
logger = logging.getLogger("prompt-api")

# FastAPI app
app = FastAPI()


# Response model
class PromptResponse(BaseModel):
    prompt: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Prompt API is running"}


def format_snapshots_to_prompt(snapshots):
    if not snapshots:
        return "No VM data available."

    timestamp = snapshots[0].get('timestamp', 'Unknown time')
    prompt = f"System Monitoring Report at {timestamp}:\n\n"

    for snap in snapshots:
        vm = snap.get('vm', 'Unknown')
        system_metrics = snap.get('system_metrics', {})
        services = snap.get('service_states', {})

        mem_total = system_metrics.get('node_memory_MemTotal_bytes', 0)
        mem_available = system_metrics.get('node_memory_MemAvailable_bytes', 0)
        mem_pct_used = (1 - mem_available / mem_total) * 100 if mem_total > 0 else 0

        prompt += f"- VM: {vm}\n"
        prompt += f"  CPU Idle: {system_metrics.get('node_cpu_seconds_total', 0):.1f} sec\n"
        prompt += f"  Memory Used: {mem_pct_used:.1f}%\n"
        prompt += f"  Failed Services: {services.get('failed_services', [])}\n"

        anomalies = snap.get('anomalies', [])
        if anomalies:
            prompt += f"  Anomalies: {', '.join(anomalies)}\n"
        prompt += "\n"

    prompt += "Suggest actions: NO_ACTION, RESTART_SERVICES, RESTART_VM, MIGRATE_VM, SCALE_UP.\n"
    return prompt


@app.get("/new_prompt", response_model=PromptResponse)
def get_prompt():
    try:
        minio = MinioService(
            endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin")
        )
        # Get the 10 most recent objects directly from MinioService
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=2)
        all_objs = minio.list_all_objects()
        objects = [obj for obj in all_objs if obj.last_modified >= cutoff]

        if not objects:
            raise ValueError("No data in MinIO from the last 2 minutes")

        parsed_data = []
        for obj in objects:
            raw = minio.get_object_data(obj.object_name)
            if raw:
                parsed_data.extend(DataParser.parse_minio_data(raw))

        if not parsed_data:
            raise ValueError("No data found in MinIO")

        critical_data = MetricsProcessor.filter_critical_data(parsed_data)
        structured_data = MetricsProcessor.structure_metrics(critical_data)

        if not structured_data:
            raise ValueError("No structured data")

        df = pd.DataFrame(structured_data)
        snapshots, _, _ = SnapshotGenerator.generate_prompts_from_df(df, window_minutes=2)

        prompt = format_snapshots_to_prompt(snapshots)
        return {"prompt": prompt}

    except Exception as e:
        logger.error(f"Prompt generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9005)