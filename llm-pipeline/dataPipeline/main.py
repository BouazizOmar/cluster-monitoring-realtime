import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from minio_service import MinioService
from data_parser import DataParser
from metrics_processor import MetricsProcessor
from snapshot_generator import SnapshotGenerator
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from cluster_instruction import generate_cluster_instruction_label
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime

# Configure logging with the exact format shown in the example
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"
)

# Create a logger with the module filename
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

def main():
    logger.info("Starting data processing pipeline")
    
    # 1. MinIO Service: list and retrieve objects
    minio_service = MinioService()
    all_objects = minio_service.list_all_objects()
    logger.info(f"Found {len(all_objects)} objects.")

    # 2. Retrieve and parse data from all objects
    parsed_data = []
    for obj in all_objects:
        data = minio_service.get_object_data(obj.object_name)
        if not data:
            continue
        parsed_data.extend(DataParser.parse_minio_data(data))
    logger.info(f"Parsed {len(parsed_data)} records.")

    # 3. Filter and structure data
    critical_data = MetricsProcessor.filter_critical_data(parsed_data)
    metrics_data = MetricsProcessor.structure_metrics(critical_data)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    logger.info(f"Processed {len(metrics_df)} metric records.")

    # 4. Generate Snapshots and Prompts
    snapshots, per_vm_prompts, multi_vm_prompts = SnapshotGenerator.generate_prompts_from_df(metrics_df, window_minutes=2)
    logger.info(f"Generated {len(snapshots)} snapshots and {len(per_vm_prompts)} per-VM prompts.")

    # 5. Feature Extraction & Normalization
    logger.info("Extracting features and normalizing data...")
    features_list = [FeatureExtractor.extract_features(snap) for snap in snapshots]
    features_array = np.vstack(features_list)

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_array)

    features_tensor = torch.tensor(features_normalized, dtype=torch.float32)
    dataset = TensorDataset(features_tensor)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 6. Initialize and train anomaly detector (autoencoder)
    logger.info("Training anomaly detection model...")
    input_dim = features_tensor.shape[1]
    anomaly_detector = AnomalyDetector(input_dim=input_dim, latent_dim=15, dropout_rate=0.2)
    anomaly_detector.scaler = scaler  # pass along the fitted scaler
    anomaly_detector.train_autoencoder(data_loader, num_epochs=300, lr=1e-3, weight_decay=1e-5)
    logger.info("Anomaly detection model training completed.")

    # 7. Compute threshold from training errors.
    anomaly_detector.compute_threshold(features_tensor)
    logger.info(f"Computed anomaly threshold: {anomaly_detector.threshold:.4f}")

    # 8. Cluster-level Instruction Generation
    logger.info("Generating cluster-level instruction dataset...")
    cluster_dataset = generate_cluster_instruction_label(snapshots, anomaly_detector, window_minutes=2)
    
    # Save the cluster dataset to a file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Add metadata to the output
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_snapshots": len(snapshots),
            "anomaly_threshold": float(anomaly_detector.threshold),
            "window_minutes": 2
        },
        "data": cluster_dataset
    }
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with timestamp for archiving
    timestamped_file = os.path.join(output_dir, f"cluster_dataset_{timestamp}.json")
    with open(timestamped_file, "w") as f:
        json.dump(output_data, f, default=str, indent=4)
    
    # Also save a standardized filename for easy access in scripts
    standard_file = "cluster_dataset.json"
    with open(standard_file, "w") as f:
        json.dump(output_data, f, default=str, indent=4)
    
    logger.info(f"Cluster dataset saved with timestamp to '{timestamped_file}'")
    logger.info(f"Cluster dataset also saved to '{standard_file}' for easy reference")
    logger.info("dataPipeline execution complete.")

if __name__ == "__main__":
    main()
