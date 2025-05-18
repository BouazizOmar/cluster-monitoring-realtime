import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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

    # Service state weights for impact calculation
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

    def analyze_service_impact(self, service_states: pd.DataFrame) -> Dict:
        """
        Analyze the impact of service states based on multiple factors.
        """
        impact_analysis = {
            'service_health': {},
            'state_transitions': {},
            'critical_services': set(),
            'degraded_services': set(),
            'impact_score': 0.0
        }

        # Group services by their current state
        state_groups = service_states.groupby('state')
        
        # Calculate base impact score
        for state, group in state_groups:
            weight = self.STATE_WEIGHTS.get(state, 0.5)
            services = group['service'].unique()
            impact_analysis['service_health'][state] = {
                'count': len(services),
                'services': services.tolist(),
                'weight': weight
            }
            impact_analysis['impact_score'] += weight * len(services)

        # Analyze state transitions
        for service in service_states['service'].unique():
            service_data = service_states[service_states['service'] == service]
            current_state = service_data['state'].iloc[-1]
            
            # Update service history
            self.service_history[service].append({
                'state': current_state,
                'timestamp': service_data['timestamp'].iloc[-1]
            })
            
            # Keep only last 10 states for pattern analysis
            if len(self.service_history[service]) > 10:
                self.service_history[service] = self.service_history[service][-10:]

            # Analyze state patterns
            if len(self.service_history[service]) >= 3:
                recent_states = [h['state'] for h in self.service_history[service][-3:]]
                if recent_states.count('failed') >= 2:
                    impact_analysis['critical_services'].add(service)
                elif recent_states.count('inactive') >= 2:
                    impact_analysis['degraded_services'].add(service)

        return impact_analysis

    def process_service_states(self, group: pd.DataFrame) -> Dict:
        """
        Process service states with advanced analysis for ML/LLM training.
        """
        services = group[group['metric'] == 'node_systemd_unit_state']
        if services.empty:
            return {
                'service_analysis': {},
                'impact_score': 0.0,
                'critical_services': [],
                'degraded_services': [],
                'state_distribution': {}
            }

        # Perform impact analysis
        impact_analysis = self.analyze_service_impact(services)
        
        # Calculate state distribution
        state_distribution = services['state'].value_counts().to_dict()
        
        # Prepare final service states output
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

    def generate_vm_snapshot(self, timestamp, vm, group):
        """Generate a snapshot (dictionary) for one VM at a given timestamp."""
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

        # Enhanced anomaly detection based on service impact
        if serv_states['impact_score'] > 5.0:  # Threshold for high impact
            snapshot['anomalies'].append('HIGH_SERVICE_IMPACT')
            if serv_states['critical_services']:
                snapshot['suggested_actions'].append('INVESTIGATE_CRITICAL_SERVICES')
            if serv_states['degraded_services']:
                snapshot['suggested_actions'].append('MONITOR_DEGRADED_SERVICES')

        # Memory usage calculation
        if sys_metrics.get('node_memory_MemTotal_bytes', 0) > 0:
            mem_pct_used = (1 - sys_metrics['node_memory_MemAvailable_bytes'] / sys_metrics['node_memory_MemTotal_bytes']) * 100
            if mem_pct_used > 90:
                snapshot['anomalies'].append('HIGH_MEMORY_USAGE')
                snapshot['suggested_actions'].append('CHECK_MEMORY')

        return snapshot
