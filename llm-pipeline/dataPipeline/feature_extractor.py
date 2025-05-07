import numpy as np

class FeatureExtractor:
    FEATURE_NAMES = [
        "CPU Idle Seconds", "CPU Guest Seconds", "CPU Online", "Disk I/O Time Seconds",
        "Disk Read Bytes", "Disk Written Bytes", "Pressure CPU Waiting Seconds",
        "Pressure Memory Waiting Seconds", "Memory Available (bytes)", "Memory Free (bytes)",
        "Memory Total (bytes)", "Swap Free (bytes)", "Swap Total (bytes)",
        "Memory Cached (bytes)", "Active Anon Memory (bytes)"
    ]

    @staticmethod
    def extract_features(snapshot):
        """Extract features from 15 critical metrics within a snapshot."""
        sm = snapshot['system_metrics']
        features = np.array([
            float(sm.get('node_cpu_seconds_total', 0)),
            float(sm.get('node_cpu_guest_seconds_total', 0)),
            float(sm.get('node_cpu_online', 0)),
            float(sm.get('node_disk_io_time_seconds_total', 0)),
            float(sm.get('node_disk_read_bytes_total', 0)),
            float(sm.get('node_disk_written_bytes_total', 0)),
            float(sm.get('node_pressure_cpu_waiting_seconds_total', 0)),
            float(sm.get('node_pressure_memory_waiting_seconds_total', 0)),
            float(sm.get('node_memory_MemAvailable_bytes', 0)),
            float(sm.get('node_memory_MemFree_bytes', 0)),
            float(sm.get('node_memory_MemTotal_bytes', 0)),
            float(sm.get('node_memory_SwapFree_bytes', 0)),
            float(sm.get('node_memory_SwapTotal_bytes', 0)),
            float(sm.get('node_memory_Cached_bytes', 0)),
            float(sm.get('node_memory_Active_anon_bytes', 0))
        ])
        return features
