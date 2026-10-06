"""
Dataset and data loader utilities for language modeling and QAD training.
Provides support for:
1. Real Instruction/SFT datasets (tatsu-lab/alpaca, HuggingFace datasets, JSONL)
2. Synthetic structured sequence generators for fast offline testing.
"""

from typing import List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader


class RealInstructionDataset(Dataset):
    """
    Real SFT/Instruction dataset (e.g. Alpaca format) tokenized for Causal LM / QAD.
    Each sample combines Instruction, Input, and Response into token IDs.
    """

    def __init__(
        self,
        samples: List[dict],
        tokenizer,
        max_length: int = 128,
    ):
        super().__init__()
        self.examples = []

        for item in samples:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")

            if inp:
                text = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{inst}\n\n### Response:\n{out}"

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            if len(input_ids) > 8:
                self.examples.append(input_ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.examples[idx]
        return seq[:-1], seq[1:]


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length token sequences in a batch."""
    max_len = max(len(x[0]) for x in batch)
    inputs = []
    targets = []
    for inp, tgt in batch:
        pad_len = max_len - len(inp)
        inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        targets.append(torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)]))
    return torch.stack(inputs), torch.stack(targets)


def get_real_dataloaders(
    tokenizer,
    dataset_name: str = "tatsu-lab/alpaca",
    max_samples: int = 1000,
    max_length: int = 128,
    batch_size: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads and tokenizes a real-world dataset (such as tatsu-lab/alpaca) for QAD training.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=f"train[:{max_samples}]")
    samples = [dict(item) for item in ds]

    num_val = max(10, int(len(samples) * val_split))
    num_train = len(samples) - num_val

    train_samples = samples[:num_train]
    val_samples = samples[num_train:]

    train_ds = RealInstructionDataset(train_samples, tokenizer, max_length=max_length)
    val_ds = RealInstructionDataset(val_samples, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, val_loader


class SyntheticLanguageDataset(Dataset):
    """
    Generates structured synthetic token sequences simulating formal grammatical patterns.
    Pattern: [Subject, Verb, Object, Modifiers, EndToken] with vocabulary dependencies,
    allowing the model to quickly learn genuine probabilistic language patterns.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 32,
        vocab_size: int = 500,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        rng = torch.Generator().manual_seed(seed)

        data = []
        for _ in range(num_samples):
            tokens = []
            curr = torch.randint(10, vocab_size // 4, (1,), generator=rng).item()
            tokens.append(curr)

            for step in range(1, seq_len):
                if step % 8 == 0:
                    next_tok = 1  # EOS or separator token
                else:
                    step_offset = (curr * 3 + step * 7) % (vocab_size - 10) + 10
                    noise = torch.randint(-2, 3, (1,), generator=rng).item()
                    next_tok = max(10, min(vocab_size - 1, step_offset + noise))

                tokens.append(next_tok)
                curr = next_tok

            data.append(torch.tensor(tokens, dtype=torch.long))

        self.data = torch.stack(data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def get_dataloaders(
    num_train: int = 1200,
    num_val: int = 300,
    seq_len: int = 32,
    vocab_size: int = 500,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Returns train and validation DataLoaders for synthetic language data."""
    train_dataset = SyntheticLanguageDataset(
        num_samples=num_train,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    val_dataset = SyntheticLanguageDataset(
        num_samples=num_val,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed + 100,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
