import os
import torch
from torch.cuda.amp import GradScaler
from mlx.optimizers import Adam
from mlx_lm.tuner import train, linear_to_lora_layers
from dataset import load_vm_dataset_from_file, SimpleDataset
from model_utils import setup_model_tokenizer, setup_adapter_and_training_args, Metrics


def main(
    data_path: str,
    model_name: str = 'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
    max_seq_length: int = 512,
    valid_frac: float = 0.1,
    learning_rate: float = 5e-6
):
    # Prepare model & tokenizer (with stable quantization)
    model, tokenizer = setup_model_tokenizer(model_name)
    model.freeze()

    # Load datasets with correct signature
    train_samples, val_samples = load_vm_dataset_from_file(
        data_path,
        valid_frac=valid_frac,
    )
    train_set = SimpleDataset(train_samples, tokenizer, max_seq_length)
    dev_set   = SimpleDataset(val_samples,   tokenizer, max_seq_length)

    print(f"Loaded {len(train_set)} train and {len(dev_set)} validation samples.")

    # Setup LoRA adapter & training args
    lora_config, training_args = setup_adapter_and_training_args(iters=50)
    linear_to_lora_layers(
        model,
        lora_config['lora_layers'],
        lora_config['lora_parameters']
    )

    # Optimizer with weight decay
    opt = Adam(learning_rate=learning_rate)

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)



    # Training
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=dev_set,
        training_callback=Metrics()
    )

    print('Fine-tuning complete.')

if __name__ == '__main__':
    # Embedded configuration
    data_path = '/datapipeline/output/cluster_dataset_2025-04-16_16-47-04.json'
    model_name = 'mlx-community/Mistral-7B-Instruct-v0.3-4bit'
    main(data_path=data_path, model_name=model_name)