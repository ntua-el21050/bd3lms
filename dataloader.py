# dataloader.py
from typing import Optional


import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


from typing import Optional, Tuple


import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from omegaconf import DictConfig


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def manual_train_valid_split(
    dataset: Dataset,
    max_train_samples: Optional[int],
    max_valid_samples: Optional[int],
    seed: int = 42,
) -> Tuple[Optional[Dataset], Dataset]:
    """
    Manually shuffle + split a dataset into train / valid subsets.
    Tokenization MUST happen after this.
    """

    dataset = dataset.shuffle(seed=seed)

    n_total = len(dataset)

    n_train = max_train_samples or (n_total * 9 // 10)
    n_valid = max_valid_samples or (n_total - n_train)

    n_train = min(n_train, n_total)
    n_valid = min(n_valid, n_total - n_train)

    train_ds = None
    if n_train > 0:
        train_ds = dataset.select(range(0, n_train))

    valid_ds = dataset.select(range(n_train, n_train + n_valid))

    return train_ds, valid_ds


def _default_text_column(dataset: Dataset) -> str:
    """
    Heuristic to find the text column automatically.
    """
    for key in ["text", "content", "sentence", "document"]:
        if key in dataset.column_names:
            return key
    raise ValueError(
        f"Could not infer text column. Available columns: {dataset.column_names}"
    )


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_raw_dataset(cfg: DictConfig) -> Dataset:
    """
    Loads the full raw dataset WITHOUT split.
    """

    if cfg.data.name is None:
        raise ValueError("cfg.data.name must be set")

    dataset = load_dataset(
        cfg.data.name,
        cfg.data.get("config", None),
        split="train",
        trust_remote_code=True,
    )

    return dataset


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def build_tokenizer(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name
        if "tokenizer_name" in cfg.model
        else cfg.model.name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    text_column: str,
    block_size: int,
    num_proc: int,
):
    """
    Tokenize + chunk into fixed-length blocks.
    """

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_column],
            truncation=False,
            add_special_tokens=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing (num_proc={num_proc})",
    )

    def group_texts(examples):
        concatenated = {}
        for k in examples.keys():
            concatenated[k] = sum(examples[k], [])

        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    return lm_ds


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

def _build_datasets(cfg: DictConfig, tokenizer):
    raw_dataset = load_raw_dataset(cfg)

    text_column = (
        cfg.data.text_column
        if "text_column" in cfg.data and cfg.data.text_column is not None
        else _default_text_column(raw_dataset)
    )

    train_raw, valid_raw = manual_train_valid_split(
        raw_dataset,
        max_train_samples=cfg.data.get("max_train_samples"),
        max_valid_samples=cfg.data.get("max_valid_samples"),
        seed=cfg.data.get("seed", 42),
    )

    num_proc = cfg.data.get("num_proc", 4)
    block_size = cfg.model.length

    train_ds = None
    if train_raw is not None:
        train_ds = tokenize_dataset(
            train_raw,
            tokenizer,
            text_column,
            block_size,
            num_proc,
        )

    valid_ds = tokenize_dataset(
        valid_raw,
        tokenizer,
        text_column,
        block_size,
        num_proc,
    )

    return train_ds, valid_ds


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

def collate_fn(batch):
    """
    Simple causal LM collator.
    """
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )


def get_dataloaders(
    config,
    tokenizer,
    skip_train=False,
    skip_valid=False,
    valid_seed=None,
):
    train_ds, valid_ds = _build_datasets(config, tokenizer)

    train_loader = None
    if not skip_train and train_ds is not None and config.mode == "train":
        train_loader = build_dataloader(
            train_ds,
            batch_size=config.loader.batch_size,
            shuffle=True,
            num_workers=config.loader.get("num_workers", 4),
        )
        train_loader.tokenizer = tokenizer

    valid_loader = None
    if not skip_valid:
        valid_loader = build_dataloader(
            valid_ds,
            batch_size=config.loader.eval_batch_size,
            shuffle=False,
            num_workers=config.loader.get("num_workers", 4),
        )
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader

def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == 'text8':
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased'
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path,
            trust_remote_code=True
        )

    # GPT-style tokenizers need BOS/EOS post processing
    if isinstance(tokenizer, (transformers.GPT2TokenizerFast, transformers.GPT2Tokenizer)):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer
