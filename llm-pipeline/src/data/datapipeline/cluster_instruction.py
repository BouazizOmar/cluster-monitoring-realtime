import json
from prompt_generation import PromptFormatter

def group_snapshots_by_window(snapshots, window_minutes=2):
    """
    Groups snapshots into windows by timestamp.
    """
    window_dict = {}
    for snap in snapshots:
        key = snap['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        window_dict.setdefault(key, []).append(snap)
    return window_dict

def generate_cluster_instruction_label(snapshots, anomaly_detector, window_minutes=2):
    """
    Generate cluster-level instruction labels for each time window.
    Each label includes smart instructions and anomaly details.
    """
    window_dict = group_snapshots_by_window(snapshots, window_minutes)
    cluster_dataset = []
    for window_key, snap_list in window_dict.items():
        # Use enhanced prompt with details.
        cluster_instruction = PromptFormatter.format_multi_vm_prompt_with_details(
            snap_list, anomaly_detector.model, anomaly_detector.scaler, anomaly_detector.threshold,
            anomaly_detector.generate_smart_instruction_label
        )
        vm_actions = {}
        vm_details = {}
        for snap in snap_list:
            record = anomaly_detector.generate_smart_instruction_label(snap)
            vm_actions[snap['vm']] = record['label']
            vm_details[snap['vm']] = {
                "anomaly_score": record["anomaly_score"],
                "anomaly_details": record["anomaly_details"]
            }
        label_data = {"actions": vm_actions, "details": vm_details}
        cluster_dataset.append({
            "instruction": cluster_instruction,
            "label": json.dumps(label_data)
        })
    return cluster_dataset
