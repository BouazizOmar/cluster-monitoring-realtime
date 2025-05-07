import json

def convert_sample(sample: dict) -> dict:
    """Convert raw sample to structured training format with validation"""
    try:
        json_label = json.loads(sample['label'])
        if 'actions' not in json_label or 'details' not in json_label:
            raise ValueError('Invalid label structure')
        return {
            'instruction': (
                'Analyze VM status report and recommend actions:\n'
                f"{sample['instruction']}\n"
                'Options: NO_ACTION, RESTART_SERVICES, RESTART_VM, MIGRATE_VM, SCALE_UP.'
            ),
            'response': sample['label']
        }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Invalid sample: {e}")
        return None

def format_text(entry: dict) -> str:
    """Create instruction-response format with special tokens"""
    return (
        "### Instruction:\n{instruction}\n\n### Response:\n{response}"
        .format(**entry)
    )