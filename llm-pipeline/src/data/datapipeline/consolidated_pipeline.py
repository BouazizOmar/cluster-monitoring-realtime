import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from minio import Minio
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"
)
logger = logging.getLogger("pipeline")


class MinioService:
    def __init__(self):
        endpoint = "localhost:9000"
        access_key = "minioadmin"
        secret_key = "minioadmin"
        bucket_name = "warehouse"
        secure = False

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name

    def list_all_objects(self, limit=None):
        all_objects = list(self.client.list_objects(self.bucket_name, recursive=True))

        # Define our time window
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        fifteen_days_ago = now - timedelta(days=15)
        ten_days_ago     = now - timedelta(days=10)

        # Filter objects between 15 and 10 days ago
        window_objects = [
            obj for obj in all_objects
            if fifteen_days_ago < obj.last_modified < ten_days_ago
        ]

        # Sort newest first (i.e. closest to 10 days ago)
        sorted_objects = sorted(
            window_objects,
            key=lambda obj: obj.last_modified,
            reverse=True
        )

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


class DataParser:
    @staticmethod
    def parse_minio_data(data):
        parsed_records = []
        for record in data:
            try:
                parsed_records.append({
                    "name": record["metric"]["__name__"],
                    "value": record["value"][1],
                    "timestamp": record["value"][0],
                    "labels": record["metric"]
                })
            except (KeyError, IndexError) as e:
                logger.warning(f"Error parsing record: {e}")
                continue
        return parsed_records

class MetricsProcessor:
    CRITICAL_METRICS = [
        "node_cpu_seconds_total",
        "node_cpu_guest_seconds_total",
        "node_cpu_online",
        "node_disk_io_time_seconds_total",
        "node_disk_read_bytes_total",
        "node_disk_written_bytes_total",
        "node_pressure_cpu_waiting_seconds_total",
        "node_pressure_memory_waiting_seconds_total",
        "node_memory_MemAvailable_bytes",
        "node_memory_MemFree_bytes",
        "node_memory_MemTotal_bytes",
        "node_memory_SwapFree_bytes",
        "node_memory_SwapTotal_bytes",
        "node_memory_Cached_bytes",
        "node_memory_Active_anon_bytes",
        "node_systemd_unit_state"
    ]

    STATE_WEIGHTS = {
        'active': 1.0,
        'inactive': 0.8,
        'failed': 0.0,
        'activating': 0.6,
        'deactivating': 0.4,
        'maintenance': 0.5
    }

    def __init__(self):
        self.service_history = defaultdict(list)
        self.service_patterns = defaultdict(dict)
        self.vm_service_roles = defaultdict(set)

    @staticmethod
    def filter_critical_data(parsed_data):
        return [obj for obj in parsed_data if obj["name"] in MetricsProcessor.CRITICAL_METRICS]

    @staticmethod
    def structure_metrics(critical_data):
        metrics_data = []
        for metric in critical_data:
            vm_name = metric['labels'].get('vm')
            metrics_data.append({
                'vm': vm_name,
                'metric': metric['name'],
                'value': float(metric['value']),
                'state': metric['labels'].get('state', None),
                'service': metric['labels'].get('name', None),
                'timestamp': metric['timestamp']
            })
        return metrics_data

    def analyze_service_impact(self, service_states: pd.DataFrame) -> Dict:
        impact_analysis = {
            'service_health': {},
            'state_transitions': {},
            'critical_services': set(),
            'degraded_services': set(),
            'impact_score': 0.0
        }

        state_groups = service_states.groupby('state')
        
        for state, group in state_groups:
            weight = self.STATE_WEIGHTS.get(state, 0.5)
            services = group['service'].unique()
            impact_analysis['service_health'][state] = {
                'count': len(services),
                'services': services.tolist(),
                'weight': weight
            }
            impact_analysis['impact_score'] += weight * len(services)

        for service in service_states['service'].unique():
            service_data = service_states[service_states['service'] == service]
            current_state = service_data['state'].iloc[-1]
            
            self.service_history[service].append({
                'state': current_state,
                'timestamp': service_data['timestamp'].iloc[-1]
            })
            
            if len(self.service_history[service]) > 10:
                self.service_history[service] = self.service_history[service][-10:]

            if len(self.service_history[service]) >= 3:
                recent_states = [h['state'] for h in self.service_history[service][-3:]]
                if recent_states.count('failed') >= 2:
                    impact_analysis['critical_services'].add(service)
                elif recent_states.count('inactive') >= 2:
                    impact_analysis['degraded_services'].add(service)

        return impact_analysis

    def process_service_states(self, group: pd.DataFrame) -> Dict:
        services = group[group['metric'] == 'node_systemd_unit_state']
        if services.empty:
            return {
                'service_analysis': {},
                'impact_score': 0.0,
                'critical_services': [],
                'degraded_services': [],
                'state_distribution': {}
            }

        impact_analysis = self.analyze_service_impact(services)
        state_distribution = services['state'].value_counts().to_dict()
        
        service_states = {
            'service_analysis': impact_analysis['service_health'],
            'impact_score': impact_analysis['impact_score'],
            'critical_services': list(impact_analysis['critical_services']),
            'degraded_services': list(impact_analysis['degraded_services']),
            'state_distribution': state_distribution,
            'state_transitions': impact_analysis['state_transitions']
        }

        return service_states

    @staticmethod
    def process_system_metrics(group):
        metrics = {}
        # CPU metrics
        metrics['node_cpu_seconds_total'] = (
            group[group['metric'] == 'node_cpu_seconds_total']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_cpu_seconds_total'].empty else 0)
        metrics['node_cpu_guest_seconds_total'] = (
            group[group['metric'] == 'node_cpu_guest_seconds_total']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_cpu_guest_seconds_total'].empty else 0)
        metrics['node_cpu_online'] = (
            group[group['metric'] == 'node_cpu_online']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_cpu_online'].empty else 0)

        # Disk metrics
        metrics['node_disk_io_time_seconds_total'] = (
            group[group['metric'] == 'node_disk_io_time_seconds_total']['value'].astype(float).sum()
            if not group[group['metric'] == 'node_disk_io_time_seconds_total'].empty else 0)
        metrics['node_disk_read_bytes_total'] = (
            group[group['metric'] == 'node_disk_read_bytes_total']['value'].astype(float).sum()
            if not group[group['metric'] == 'node_disk_read_bytes_total'].empty else 0)
        metrics['node_disk_written_bytes_total'] = (
            group[group['metric'] == 'node_disk_written_bytes_total']['value'].astype(float).sum()
            if not group[group['metric'] == 'node_disk_written_bytes_total'].empty else 0)

        # Pressure metrics
        metrics['node_pressure_cpu_waiting_seconds_total'] = (
            group[group['metric'] == 'node_pressure_cpu_waiting_seconds_total']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_pressure_cpu_waiting_seconds_total'].empty else 0)
        metrics['node_pressure_memory_waiting_seconds_total'] = (
            group[group['metric'] == 'node_pressure_memory_waiting_seconds_total']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_pressure_memory_waiting_seconds_total'].empty else 0)

        # Memory metrics
        metrics['node_memory_MemAvailable_bytes'] = (
            group[group['metric'] == 'node_memory_MemAvailable_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_MemAvailable_bytes'].empty else 0)
        metrics['node_memory_MemFree_bytes'] = (
            group[group['metric'] == 'node_memory_MemFree_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_MemFree_bytes'].empty else 0)
        metrics['node_memory_MemTotal_bytes'] = (
            group[group['metric'] == 'node_memory_MemTotal_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_MemTotal_bytes'].empty else 0)
        metrics['node_memory_SwapFree_bytes'] = (
            group[group['metric'] == 'node_memory_SwapFree_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_SwapFree_bytes'].empty else 0)
        metrics['node_memory_SwapTotal_bytes'] = (
            group[group['metric'] == 'node_memory_SwapTotal_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_SwapTotal_bytes'].empty else 0)
        metrics['node_memory_Cached_bytes'] = (
            group[group['metric'] == 'node_memory_Cached_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_Cached_bytes'].empty else 0)
        metrics['node_memory_Active_anon_bytes'] = (
            group[group['metric'] == 'node_memory_Active_anon_bytes']['value'].astype(float).mean()
            if not group[group['metric'] == 'node_memory_Active_anon_bytes'].empty else 0)

        return metrics

    def generate_vm_snapshot(self, timestamp, vm, group):
        sys_metrics = self.process_system_metrics(group)
        serv_states = self.process_service_states(group)
        
        snapshot = {
            'timestamp': timestamp,
            'vm': vm,
            'instance': group['labels.instance'].iloc[0] if 'labels.instance' in group.columns else vm,
            'system_metrics': sys_metrics,
            'service_states': serv_states,
            'anomalies': [],
            'suggested_actions': []
        }

        if serv_states['impact_score'] > 5.0:
            snapshot['anomalies'].append('HIGH_SERVICE_IMPACT')
            if serv_states['critical_services']:
                snapshot['suggested_actions'].append('INVESTIGATE_CRITICAL_SERVICES')
            if serv_states['degraded_services']:
                snapshot['suggested_actions'].append('MONITOR_DEGRADED_SERVICES')

        if sys_metrics.get('node_memory_MemTotal_bytes', 0) > 0:
            mem_pct_used = (1 - sys_metrics['node_memory_MemAvailable_bytes'] / sys_metrics['node_memory_MemTotal_bytes']) * 100
            if mem_pct_used > 90:
                snapshot['anomalies'].append('HIGH_MEMORY_USAGE')
                snapshot['suggested_actions'].append('CHECK_MEMORY')

        return snapshot

class SnapshotGenerator:
    def __init__(self):
        self.metrics_processor = MetricsProcessor()

    def generate_prompts_from_df(self, df, window_minutes=5):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        snapshots = []
        per_vm_prompts = []
        multi_vm_prompts = {}

        grouped = df.groupby([pd.Grouper(key='timestamp', freq=f'{window_minutes}Min'), 'vm'])
        for (timestamp, vm), group in grouped:
            snap = self.metrics_processor.generate_vm_snapshot(timestamp, vm, group)
            snapshots.append(snap)
            per_vm_prompts.append(self.format_llm_prompt(snap))
            window_key = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            multi_vm_prompts.setdefault(window_key, []).append(snap)

        multi_vm_prompt_list = [(window_key, snap_list) for window_key, snap_list in multi_vm_prompts.items()]
        return snapshots, per_vm_prompts, multi_vm_prompt_list

    @staticmethod
    def format_llm_prompt(snapshot):
        prompt = f"VM State Analysis for {snapshot['vm']} at {snapshot['timestamp']}\n\n"
        
        # System metrics
        prompt += "System Metrics:\n"
        for metric, value in snapshot['system_metrics'].items():
            prompt += f"- {metric}: {value}\n"
        
        # Service states
        prompt += "\nService States:\n"
        for state, info in snapshot['service_states']['service_analysis'].items():
            prompt += f"- {state}: {info['count']} services\n"
        
        # Anomalies
        if snapshot['anomalies']:
            prompt += "\nDetected Anomalies:\n"
            for anomaly in snapshot['anomalies']:
                prompt += f"- {anomaly}\n"
        
        # Suggested actions
        if snapshot['suggested_actions']:
            prompt += "\nSuggested Actions:\n"
            for action in snapshot['suggested_actions']:
                prompt += f"- {action}\n"
        
        return prompt

class FeatureExtractor:
    @staticmethod
    def extract_features(snapshot):
        features = []
        
        # System metrics
        sys_metrics = snapshot['system_metrics']
        features.extend([
            sys_metrics.get('node_cpu_seconds_total', 0),
            sys_metrics.get('node_cpu_guest_seconds_total', 0),
            sys_metrics.get('node_cpu_online', 0),
            sys_metrics.get('node_disk_io_time_seconds_total', 0),
            sys_metrics.get('node_disk_read_bytes_total', 0),
            sys_metrics.get('node_disk_written_bytes_total', 0),
            sys_metrics.get('node_pressure_cpu_waiting_seconds_total', 0),
            sys_metrics.get('node_pressure_memory_waiting_seconds_total', 0),
            sys_metrics.get('node_memory_MemAvailable_bytes', 0),
            sys_metrics.get('node_memory_MemFree_bytes', 0),
            sys_metrics.get('node_memory_MemTotal_bytes', 0),
            sys_metrics.get('node_memory_SwapFree_bytes', 0),
            sys_metrics.get('node_memory_SwapTotal_bytes', 0),
            sys_metrics.get('node_memory_Cached_bytes', 0),
            sys_metrics.get('node_memory_Active_anon_bytes', 0)
        ])
        
        # Service states
        serv_states = snapshot['service_states']
        features.extend([
            serv_states.get('impact_score', 0),
            len(serv_states.get('critical_services', [])),
            len(serv_states.get('degraded_services', []))
        ])
        
        return np.array(features, dtype=np.float32)

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim, latent_dim=15, dropout_rate=0.2):
        super(AnomalyDetector, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim)
        )
        
        self.threshold = None
        self.scaler = None

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_autoencoder(self, data_loader, num_epochs=300, lr=1e-3, weight_decay=1e-5):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in data_loader:
                x = batch[0]
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}')

    def compute_threshold(self, data, percentile=95):
        self.eval()
        with torch.no_grad():
            reconstructions = self(data)
            mse = torch.mean((data - reconstructions) ** 2, dim=1)
            self.threshold = torch.quantile(mse, percentile/100)

    def detect_anomaly(self, data):
        self.eval()
        with torch.no_grad():
            reconstructions = self(data)
            mse = torch.mean((data - reconstructions) ** 2, dim=1)
            return mse > self.threshold

def generate_cluster_instruction_label(snapshots, anomaly_detector, window_minutes=2):
    cluster_dataset = []
    
    for snapshot in snapshots:
        features = FeatureExtractor.extract_features(snapshot)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        is_anomaly = anomaly_detector.detect_anomaly(features_tensor).item()
        
        instruction = {
            'timestamp': snapshot['timestamp'],
            'vm': snapshot['vm'],
            'is_anomaly': bool(is_anomaly),
            'anomalies': snapshot['anomalies'],
            'suggested_actions': snapshot['suggested_actions'],
            'system_metrics': snapshot['system_metrics'],
            'service_states': snapshot['service_states']
        }
        
        cluster_dataset.append(instruction)
    
    return cluster_dataset

def run_pipeline(window_minutes=2):
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
    snapshot_generator = SnapshotGenerator()
    snapshots, per_vm_prompts, multi_vm_prompts = snapshot_generator.generate_prompts_from_df(metrics_df, window_minutes=window_minutes)
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

    # 6. Initialize and train anomaly detector
    logger.info("Training anomaly detection model...")
    input_dim = features_tensor.shape[1]
    anomaly_detector = AnomalyDetector(input_dim=input_dim, latent_dim=15, dropout_rate=0.2)
    anomaly_detector.scaler = scaler
    anomaly_detector.train_autoencoder(data_loader, num_epochs=300, lr=1e-3, weight_decay=1e-5)
    logger.info("Anomaly detection model training completed.")

    # 7. Compute threshold
    anomaly_detector.compute_threshold(features_tensor)
    logger.info(f"Computed anomaly threshold: {anomaly_detector.threshold:.4f}")

    # 8. Generate cluster dataset
    logger.info("Generating cluster-level instruction dataset...")
    cluster_dataset = generate_cluster_instruction_label(snapshots, anomaly_detector, window_minutes=window_minutes)
    
    # Save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_snapshots": len(snapshots),
            "anomaly_threshold": float(anomaly_detector.threshold),
            "window_minutes": window_minutes
        },
        "data": cluster_dataset
    }
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamped_file = os.path.join(output_dir, f"cluster_dataset_{timestamp}.json")
    with open(timestamped_file, "w") as f:
        json.dump(output_data, f, default=str, indent=4)
    
    standard_file = "cluster_dataset.json"
    with open(standard_file, "w") as f:
        json.dump(output_data, f, default=str, indent=4)
    
    logger.info(f"Cluster dataset saved with timestamp to '{timestamped_file}'")
    logger.info(f"Cluster dataset also saved to '{standard_file}' for easy reference")
    logger.info("Pipeline execution complete.")
    
    return {
        'snapshots': snapshots,
        'per_vm_prompts': per_vm_prompts,
        'multi_vm_prompts': multi_vm_prompts,
        'cluster_dataset': cluster_dataset,
        'anomaly_detector': anomaly_detector,
        'scaler': scaler
    }

if __name__ == "__main__":
    run_pipeline() 