
import logging

logging.basicConfig(level=logging.INFO)

FEATURE_NAMES = [
    "CPU Idle Seconds", "CPU Guest Seconds", "CPU Online", "Disk I/O Time Seconds",
    "Disk Read Bytes", "Disk Written Bytes", "Pressure CPU Waiting Seconds",
    "Pressure Memory Waiting Seconds", "Memory Available (bytes)", "Memory Free (bytes)",
    "Memory Total (bytes)", "Swap Free (bytes)", "Swap Total (bytes)",
    "Memory Cached (bytes)", "Active Anon Memory (bytes)"
]

class PromptFormatter:
    @staticmethod
    def format_llm_prompt(snapshot):
        """Create a human-friendly prompt for a single VM."""
        vm = snapshot['vm']
        ts = snapshot['timestamp']
        inst = snapshot.get('instance', vm)
        sm = snapshot['system_metrics']
        ss = snapshot['service_states']
        if sm.get('node_memory_MemTotal_bytes', 0) > 0:
            mem_pct_used = (1 - sm.get('node_memory_MemAvailable_bytes', 0) / sm.get('node_memory_MemTotal_bytes', 1)) * 100
            mem_total_gb = sm.get('node_memory_MemTotal_bytes', 0) / 1e9
            mem_available_gb = sm.get('node_memory_MemAvailable_bytes', 0) / 1e9
        else:
            mem_pct_used, mem_total_gb, mem_available_gb = 0, 0, 0

        prompt = f"System Monitoring Report for VM {vm} (Instance: {inst}) at {ts}:\n"
        prompt += f"- CPU Idle: {sm.get('node_cpu_seconds_total', 0):.1f} seconds\n"
        prompt += f"- Memory used: {mem_pct_used:.1f}% of {mem_total_gb:.1f}GB total (Available: {mem_available_gb:.1f}GB)\n"
        prompt += f"- Disk I/O time: {sm.get('node_disk_io_time_seconds_total', 0):.1f} seconds\n"
        prompt += f"- Failed services: {ss.get('failed_services', [])} (Count: {ss.get('failed_count', 0)})\n"
        if snapshot['anomalies']:
            prompt += f"Detected anomalies: {', '.join(snapshot['anomalies'])}\n"
        prompt += ("\nBased on the above data, what action should be taken for this VM? "
                   "Options: NO_ACTION, RESTART_SERVICES, RESTART_VM, MIGRATE_VM, SCALE_UP. Explain your reasoning.")
        return prompt

    @staticmethod
    def format_multi_vm_prompt_with_details(snapshots, model, scaler, threshold, smart_label_func):
        """
        Create a cluster-level instruction prompt, appending anomaly details for each VM.
        The smart_label_func is used to get anomaly details per snapshot.
        """
        if not snapshots:
            return ""

        ts = snapshots[0]['timestamp']
        prompt = f"System Monitoring Report at {ts} across VMs:\n\n"

        for snap in snapshots:
            vm = snap['vm']
            inst = snap.get('instance', vm)
            sm = snap['system_metrics']
            ss = snap['service_states']

            if sm.get('node_memory_MemTotal_bytes', 0) > 0:
                mem_pct_used = (1 - sm.get('node_memory_MemAvailable_bytes', 0) /
                                sm.get('node_memory_MemTotal_bytes', 1)) * 100
                mem_total_gb = sm.get('node_memory_MemTotal_bytes', 0) / 1e9
                mem_available_gb = sm.get('node_memory_MemAvailable_bytes', 0) / 1e9
            else:
                mem_pct_used, mem_total_gb, mem_available_gb = 0, 0, 0

            vm_prompt = f"VM {vm} (Instance: {inst}):\n"
            vm_prompt += f"  - CPU Idle: {sm.get('node_cpu_seconds_total', 0):.1f} seconds\n"
            vm_prompt += f"  - Memory used: {mem_pct_used:.1f}% of {mem_total_gb:.1f}GB total (Available: {mem_available_gb:.1f}GB)\n"
            vm_prompt += f"  - Disk I/O time: {sm.get('node_disk_io_time_seconds_total', 0):.1f} seconds\n"
            vm_prompt += f"  - Failed services: {ss.get('failed_services', [])} (Count: {ss.get('failed_count', 0)})\n"

            if snap['anomalies']:
                vm_prompt += f"  Detected anomalies: {', '.join(snap['anomalies'])}\n"

            # Call the smart label function with only the snapshot
            record = smart_label_func(snap)

            if record["anomaly_details"]:
                details_text = "; ".join(record["anomaly_details"])
                vm_prompt += f"  Anomaly Details: {details_text}\n"

            vm_prompt += "\n"
            prompt += vm_prompt

        prompt += (
            "Based on the above data, compare the statuses of the VMs and decide the appropriate action for each. "
            "Options: NO_ACTION, RESTART_SERVICES, RESTART_VM, MIGRATE_VM, SCALE_UP. "
            "Explain your reasoning for each VM.")

        return prompt

