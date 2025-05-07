import json
from pathlib import Path
from transformers import AutoTokenizer
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs


def setup_model_tokenizer(model_name: str, modified_dir: str = 'modified_tokenizer'):
    """Initialize tokenizer with special tokens, save & reload; return model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'additional_special_tokens': [
            '### Instruction:', '### Response:',
            '[VM]', '[CPU]', '[MEMORY]', '[DISK]'
        ]
    })
    tokenizer.add_tokens([
        'NO_ACTION', 'RESTART_SERVICES',
        'RESTART_VM', 'MIGRATE_VM', 'SCALE_UP'
    ])
    tokenizer.save_pretrained(modified_dir)

    model, _ = load(model_name, tokenizer_config={})

    modified_tokenizer = AutoTokenizer.from_pretrained(modified_dir, use_fast=False)
    return model, modified_tokenizer


def setup_adapter_and_training_args(
    adapter_dir: str = 'adapters',
    num_layers: int = 8,
    lora_layers: int = 8,
    rank: int = 8,
    scale: float = 20.0,
    dropout: float = 0.1,
    iters: int = 50,
):
    """Prepare and save LoRA adapter config and TrainingArgs."""
    adapter_path = Path(adapter_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    lora_config = {
        'num_layers': num_layers,
        'lora_layers': lora_layers,
        'lora_parameters': {
            'rank': rank,
            'scale': scale,
            'dropout': dropout,
        }
    }
    with open(adapter_path / 'adapter_config.json', 'w') as f:
        json.dump(lora_config, f, indent=4)

    training_args = TrainingArgs(
        adapter_file=adapter_path / 'adapters.safetensors',
        iters=iters,
        steps_per_eval=10,
    )
    return lora_config, training_args


class Metrics:
    """Callback for tracking training and validation losses."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_loss_report(self, info):
        self.train_losses.append((info['iteration'], info['train_loss']))

    def on_val_loss_report(self, info):
        self.val_losses.append((info['iteration'], info['val_loss']))