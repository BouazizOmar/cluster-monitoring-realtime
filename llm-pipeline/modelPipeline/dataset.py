import json
import random
from typing import Tuple, List
from transformers import PreTrainedTokenizer
from data_processing import convert_sample, format_text


def load_vm_dataset_from_file(
    file_path: str,
    valid_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Load a JSON file, convert & format each sample, then shuffle & split into train/valid/test.
    """
    with open(file_path, 'r') as f:
        raw = json.load(f)
    records = raw.get('data', [])
    if not records:
        raise ValueError(f"No 'data' array found in {file_path}")

    processed = [
        format_text(conv)
        for entry in records
        if (conv := convert_sample(entry))
    ]
    if not processed:
        raise ValueError('No valid samples found after conversion!')

    random.seed(seed)
    random.shuffle(processed)
    n = len(processed)
    n_valid = int(n * valid_frac)

    valid = processed[:n_valid]
    train = processed[n_valid :]

    assert train, 'Train split is empty!'
    return train, valid


class SimpleDataset:
    """
    A simple dataset wrapper for raw text samples; tokenizes & pads/truncates to max_length.
    """

    def __init__(self, data, tokenizer: PreTrainedTokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process(self, sample: str, idx: int = None):
        encoding = self.tokenizer(
            sample,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        # Sanity-check: no all-pad inputs
        assert encoding['input_ids'].sum() > 0, f"Empty or all-pad sample at idx {idx}"
        return encoding['input_ids'][0].tolist()

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)