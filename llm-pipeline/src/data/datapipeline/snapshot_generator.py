import pandas as pd
from metrics_processor import MetricsProcessor
from prompt_generation import PromptFormatter


class SnapshotGenerator:
    def __init__(self):
        self.metrics_processor = MetricsProcessor()

    def generate_prompts_from_df(self, df, window_minutes=5):
        """
        Given a DataFrame of metric data, generate snapshots and corresponding prompts.
        Returns:
          - snapshots: list of snapshot dictionaries per VM per time window,
          - per_vm_prompts: list of one-prompt-per-VM,
          - multi_vm_prompt_list: a list of tuples (time window key, snapshots in that window)
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        snapshots = []
        per_vm_prompts = []
        multi_vm_prompts = {}

        grouped = df.groupby([pd.Grouper(key='timestamp', freq=f'{window_minutes}Min'), 'vm'])
        for (timestamp, vm), group in grouped:
            snap = self.metrics_processor.generate_vm_snapshot(timestamp, vm, group)
            snapshots.append(snap)
            per_vm_prompts.append(PromptFormatter.format_llm_prompt(snap))
            window_key = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            multi_vm_prompts.setdefault(window_key, []).append(snap)

        multi_vm_prompt_list = [(window_key, snap_list) for window_key, snap_list in multi_vm_prompts.items()]
        return snapshots, per_vm_prompts, multi_vm_prompt_list
