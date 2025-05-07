import pandas as pd

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

    @staticmethod
    def filter_critical_data(parsed_data):
        """
        Filter parsed records to keep only those corresponding to critical metrics.
        """
        return [obj for obj in parsed_data if obj["name"] in MetricsProcessor.CRITICAL_METRICS]

    @staticmethod
    def structure_metrics(critical_data):
        """
        Convert critical metric records into a standardized dict format.
        """
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

    @staticmethod
    def process_system_metrics(group):
        """Extract various system metrics from the grouped DataFrame."""
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

    @staticmethod
    def process_service_states(group):
        """Extract and aggregate service state metrics from the grouped DataFrame."""
        CRITICAL_SERVICES = {
            "sshd.service", "systemd-journald.service", "systemd-logind.service",
            "NetworkManager.service", "nginx.service"
        }
        services = group[group['metric'] == 'node_systemd_unit_state']
        failed = services[services['state'] == 'failed']
        critical_failed = failed[failed['service'].isin(CRITICAL_SERVICES)]
        service_states = {
            'active_count': int(services[services['state'] == 'active']['value'].astype(float).sum())
            if not services.empty else 0,
            'failed_count': len(critical_failed['service'].unique()) if not critical_failed.empty else 0,
            'failed_services': critical_failed['service'].unique().tolist() if not critical_failed.empty else []
        }
        if len(service_states['failed_services']) > 3:
            service_states['failed_services'] = service_states['failed_services'][:5] + ['...']
        return service_states

    @staticmethod
    def generate_vm_snapshot(timestamp, vm, group):
        """Generate a snapshot (dictionary) for one VM at a given timestamp."""
        sys_metrics = MetricsProcessor.process_system_metrics(group)
        serv_states = MetricsProcessor.process_service_states(group)
        snapshot = {
            'timestamp': timestamp,
            'vm': vm,
            'instance': group['labels.instance'].iloc[0] if 'labels.instance' in group.columns else vm,
            'system_metrics': sys_metrics,
            'service_states': serv_states,
            'anomalies': [],
            'suggested_actions': []
        }
        # Memory usage calculation and anomaly detection logic
        if sys_metrics.get('node_memory_MemTotal_bytes', 0) > 0:
            mem_pct_used = (1 - sys_metrics['node_memory_MemAvailable_bytes'] / sys_metrics['node_memory_MemTotal_bytes']) * 100
        else:
            mem_pct_used = 0
        if mem_pct_used > 90:
            snapshot['anomalies'].append('HIGH_MEMORY_USAGE')
            snapshot['suggested_actions'].append('CHECK_MEMORY')
        if serv_states.get('failed_count', 0) > 0:
            snapshot['anomalies'].append('FAILED_SERVICES')
            snapshot['suggested_actions'].append('RESTART_SERVICES')
        return snapshot
