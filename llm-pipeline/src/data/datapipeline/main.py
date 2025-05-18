import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from minio_service import MinioService
from data_parser import DataParser
from metrics_processor import MetricsProcessor
from snapshot_generator import SnapshotGenerator
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from cluster_instruction import generate_cluster_instruction_label

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"
)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    logger.info("Starting data processing pipeline")

    # 1. List & fetch from MinIO
    svc = MinioService()
    objs = svc.list_all_objects()
    logger.info(f"Found {len(objs)} objects in bucket '{svc.bucket_name}'.")

    # 2. Parse JSON → flat records
    parsed = []
    for obj in objs:
        data = svc.get_object_data(obj.object_name)
        if data is None:
            continue
        # DataParser expects a list of Prometheus records
        parsed.extend(DataParser.parse_minio_data(data))
    logger.info(f"Parsed {len(parsed)} records.")

    # 3. Filter & structure
    crit  = MetricsProcessor.filter_critical_data(parsed)
    md    = MetricsProcessor.structure_metrics(crit)
    df    = pd.DataFrame(md)
    logger.info(f"Structured into {len(df)} metric rows.")

    # 4. Snapshots & prompts
    sg = SnapshotGenerator()
    snaps, vm_prompts, multi_prompts = sg.generate_prompts_from_df(df, window_minutes=2)
    logger.info(f"Generated {len(snaps)} snapshots and {len(vm_prompts)} prompts.")

    # 5. Feature extraction & normalization
    feats = np.vstack([FeatureExtractor.extract_features(s) for s in snaps])
    scaler = StandardScaler().fit(feats)
    norm   = scaler.transform(feats)
    tensor = torch.tensor(norm, dtype=torch.float32)
    nan_mask = torch.isnan(tensor)
    col_sum = torch.nansum(tensor, dim=0)
    col_count = (~nan_mask).sum(dim=0)
    col_mean = col_sum / col_count
    filled = torch.where(
        nan_mask,
        col_mean.unsqueeze(0).expand_as(tensor),
        tensor
    )
    # Sanity check: no NaNs remain
    assert torch.isnan(filled).sum().item() == 0, "There are still NaNs in your data!"

    dataset = TensorDataset(filled)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 6. Train anomaly detector
    logger.info("Training anomaly detector...")
    model = AnomalyDetector(input_dim=tensor.shape[1], latent_dim=15, dropout_rate=0.2)
    model.scaler = scaler
    model.train_autoencoder(loader, num_epochs=300, lr=1e-3, weight_decay=1e-5)
    model.compute_threshold(tensor)
    logger.info(f"Anomaly threshold set to {model.threshold:.4f}")

    # 7. Cluster‐level instructions
    logger.info("Generating cluster instruction dataset...")
    cluster_data = generate_cluster_instruction_label(snaps, model, window_minutes=2)

    # 8. Persist output
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_snapshots": len(snaps),
            "anomaly_threshold": float(model.threshold),
            "window_minutes": 2
        },
        "data": cluster_data
    }
    os.makedirs("output", exist_ok=True)
    path_ts = f"output/cluster_dataset_{ts}.json"
    with open(path_ts, "w") as f:
        json.dump(out, f, default=str, indent=4)
    with open("cluster_dataset.json", "w") as f:
        json.dump(out, f, default=str, indent=4)

    logger.info(f"Saved timestamped output to '{path_ts}'")
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
