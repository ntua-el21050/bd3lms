# dataloader.py
from typing import Optional

import math
import typing
import os
import shutil
import urllib
import zipfile
import tokenizers
import torch
import transformers
import re
import datasets

import functools
import itertools
import json
import requests

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

import os
import shutil
import urllib
import zipfile
import fsspec


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token='[BOS]',
        eos_token='[EOS]',
        sep_token='[SEP]',
        cls_token='[CLS]',
        pad_token='[PAD]',
        mask_token='[MASK]',
        unk_token='[UNK]',
        **kwargs):
        self.characters = list('abcdefghijklmnopqrstuvwxyz ')
        self._vocab_str_to_int = {
            '[CLS]': 0,
            '[SEP]': 1,
            '[BOS]': 2,
            '[EOS]': 3,
            '[MASK]': 4,
            '[PAD]': 5,
            '[RESERVED]': 6,
            '[UNK]': 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)}}
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(
            token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int



def get_text8_dataset(cache_dir, max_seq_length=256,
                      drop_last=True, crop_train=False):
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
      drop_last: bool, whether to drop the last incomplete batch.
      crop_train: bool, whether to subsample contiguous subsequences.

    Returns:
      dataset: datasets.DatasetDict, with keys 'train', 'validation', 'test'.
    """
    url = 'http://mattmahoney.net/dc/text8.zip'
    if not crop_train:
        cache_dir = f'{cache_dir}/text8'
    else:
        cache_dir = f'{cache_dir}/text8-crop-train'
    
    split_names = ['train', 'validation', 'test']
    
    # Check if processed dataset exists
    if not all([
        os.path.exists(os.path.join(cache_dir, split))
        for split in split_names
    ]):
        # Check if raw data exists
        raw_cache_dir = os.path.join(cache_dir, 'raw_data')
        if not all([
            os.path.exists(os.path.join(raw_cache_dir, f'text8.{split}.txt'))
            for split in split_names
        ]):
            # Download and extract
            if not os.path.exists(os.path.join(raw_cache_dir, 'text8.zip')):
                os.makedirs(raw_cache_dir, exist_ok=True)
                LOGGER.info('Downloading text8 from URL {}.'.format(url))
                with (urllib.request.urlopen(url) as in_stream,
                      open(os.path.join(raw_cache_dir, 'text8.zip'),
                           'wb') as out_file):
                    shutil.copyfileobj(in_stream, out_file)

            with open(os.path.join(raw_cache_dir, 'text8.zip'), 'rb') as f:
                rawdata = zipfile.ZipFile(f).read('text8').decode('utf-8')

            # Splits taken from D3PM codebase
            splits = {
                'train': rawdata[:90000000],
                'validation': rawdata[90000000: 95000000],
                'test': rawdata[95000000:],
            }

            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f'text8.{split}.txt')
                with open(_path, 'w') as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f'text8.{split}.txt')
                with open(_path, 'r') as f:
                    splits[split] = f.read()

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == 'train' and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = Dataset.from_dict({'text': text})
        
        from datasets import DatasetDict
        dataset = DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        from datasets import load_from_disk
        dataset = load_from_disk(cache_dir)

    return dataset


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

def load_raw_dataset(cfg: DictConfig):
    """
    Loads the full raw dataset WITHOUT split.
    Resolves bd3lms dataset aliases to HF datasets.
    """
    dataset_name = (
        cfg.data.train if cfg.mode == "train" else cfg.data.valid
    )

    if dataset_name is None:
        raise ValueError("Dataset name must be set in cfg.data.train or cfg.data.valid")

    # Dataset name resolution
    if dataset_name == "wikitext103":
        dataset = load_dataset(
            'wikitext',
            name='wikitext-103-raw-v1',
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']  # Return only train split for manual splitting
    
    elif dataset_name == "wikitext2":
        dataset = load_dataset(
            'wikitext',
            name='wikitext-2-raw-v1',
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']
    
    elif dataset_name == "ptb":
        dataset = load_dataset(
            'ptb_text_only',
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']
    
    elif dataset_name == "lambada":
        # lambada returns a single dataset, not a DatasetDict
        return get_lambada_test_dataset()
    
    elif dataset_name == "text8":
        block_size = cfg.model.length
        dataset_dict = get_text8_dataset(
            cfg.data.cache_dir, 
            max_seq_length=block_size,
            crop_train=False
        )
        return dataset_dict  # Returns DatasetDict
    
    elif dataset_name == "text8-crop":
        block_size = cfg.model.length
        dataset_dict = get_text8_dataset(
            cfg.data.cache_dir, 
            max_seq_length=block_size,
            crop_train=True
        )
        return dataset_dict
    
    elif dataset_name == 'openwebtext-train':
        dataset = load_dataset(
            'openwebtext',
            split='train[:-100000]',
            cache_dir=cfg.data.cache_dir,
            trust_remote_code=True
        )
        return dataset
    
    elif dataset_name == 'openwebtext-valid':
        dataset = load_dataset(
            'openwebtext',
            split='train[-100000:]',
            cache_dir=cfg.data.cache_dir,
            trust_remote_code=True
        )
        return dataset
    
    elif dataset_name == 'scientific_papers_arxiv':
        dataset = load_dataset(
            'scientific_papers', 'arxiv',
            trust_remote_code=True,
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']
    
    elif dataset_name == 'scientific_papers_pubmed':
        dataset = load_dataset(
            'scientific_papers', 'pubmed',
            trust_remote_code=True,
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']
    
    elif dataset_name == 'ag_news':
        dataset = load_dataset(
            'ag_news',
            cache_dir=cfg.data.cache_dir,
        )
        return dataset['train']
    
    else:
        # Generic dataset loading
        dataset = load_dataset(
            dataset_name,
            cache_dir=cfg.data.cache_dir,
            trust_remote_code=True,
        )
        # Try to get train split if it exists
        if hasattr(dataset, 'keys') and 'train' in dataset.keys():
            return dataset['train']
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



def wt_detokenizer(string):
  # contractions
  string = string.replace("s '", "s'")
  string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
  # number separators
  string = string.replace(" @-@ ", "-")
  string = string.replace(" @,@ ", ",")
  string = string.replace(" @.@ ", ".")
  # punctuation
  string = string.replace(" : ", ": ")
  string = string.replace(" ; ", "; ")
  string = string.replace(" . ", ". ")
  string = string.replace(" ! ", "! ")
  string = string.replace(" ? ", "? ")
  string = string.replace(" , ", ", ")
  # double brackets
  string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
  string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
  string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
  string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
  string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
  # miscellaneous
  string = string.replace("= = = =", "====")
  string = string.replace("= = =", "===")
  string = string.replace("= =", "==")
  string = string.replace(" " + chr(176) + " ", chr(176))
  string = string.replace(" \n", "\n")
  string = string.replace("\n ", "\n")
  string = string.replace(" N ", " 1 ")
  string = string.replace(" 's", "'s")
  return string


def ptb_detokenizer(x):
  x = x.replace(" 's", "'s")
  x = x.replace("s ' ", "s' ")
  x = x.replace(" n't", "n't")
  x = x.replace(" \n ", "\n")
  x = x.replace("\\/", "/")
  for _ in range(10):
      x = x.replace(" N ", " 1 ")
  x = x.replace("$ 1", "$1")
  x = x.replace("# 1", "#1")
  x = x.replace("<unk>", "?")
  return x


def lm1b_detokenizer(x):
  x = x.replace('http : / / ', 'http://')
  x = x.replace('https : / / ', 'https://')
  x = re.sub(r' \'(\w+)', r"'\1", x)
  x = re.sub(r' (\w+) \. ', r' \1. ', x)
  x = re.sub(r' (\w+) \.$', r' \1.', x)
  x = x.replace(' ? ', '? ')
  x = re.sub(r' \?$', '?', x)
  x = x.replace(' ! ', '! ')
  x = re.sub(r' \!$', '!', x)
  x = x.replace(' , ', ', ')
  x = x.replace(' : ', ': ')
  x = x.replace(' ; ', '; ')
  x = x.replace(' / ', '/')
  x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
  x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
  x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
  x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
  x = x.replace('$ ', '$')
  x = x.replace('£ ', '£')
  return x


def lambada_detokenizer(text):
  text = text.replace("“", '"')
  text = text.replace("”", '"')
  return '\n'+text.strip()


def scientific_papers_detokenizer(x):
  x = wt_detokenizer(x)
  x = lm1b_detokenizer(x)
  return x


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
      response = requests.get(url, stream=True)
      data_list = []

      # Process each line in the response content
      for line in response.iter_lines(decode_unicode=True):
        if line:
          data = json.loads(line)
          data_list.append(data)

      return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def _group_texts(examples, block_size, bos, eos, insert_special_tokens=True):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  if insert_special_tokens:
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  else:
    new_block_size = block_size
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    if insert_special_tokens:
      _values.append(
        [bos]
        + concatenated_examples[i : i + new_block_size]
        + [eos])
    else:
      _values.append(
        concatenated_examples[i : i + new_block_size]
      )
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result



def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    text_column: str,
    block_size: int,
    num_proc: int,
    dataset_name: str = None,
    wrap: bool = True,
    insert_eos: bool = True,
    insert_special_tokens: bool = True,
):
    """
    Tokenize + chunk into fixed-length blocks with detokenization support.
    """
    
    # Determine detokenizer based on dataset name
    if dataset_name is not None:
        if dataset_name.startswith('wikitext'):
            detokenizer = wt_detokenizer
        elif dataset_name == 'ptb':
            detokenizer = ptb_detokenizer
        elif dataset_name == 'lm1b':
            detokenizer = lm1b_detokenizer
        elif dataset_name == 'lambada':
            detokenizer = lambada_detokenizer
        elif dataset_name.startswith('scientific_papers'):
            detokenizer = scientific_papers_detokenizer
        else:
            detokenizer = None
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text
        return detok

    EOS = tokenizer.encode(tokenizer.eos_token)[0]
    BOS = tokenizer.encode(tokenizer.bos_token)[0]

    def preprocess_and_tokenize(example):
        text = example[text_column]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        if wrap:
            tokens = tokenizer(text,
                             add_special_tokens=False,
                             return_attention_mask=False,
                             return_token_type_ids=False)
            if insert_eos:
                tokens = {'input_ids':
                          [t + [EOS] for t in tokens['input_ids']]}
            # BOS will be added in group_texts
        elif dataset_name == 'lambada':
            tokens = tokenizer(text,
                             truncation=True,
                             add_special_tokens=True,
                             return_attention_mask=True,
                             return_token_type_ids=True)
        else:
            tokens = tokenizer(text,
                             max_length=block_size,
                             padding='max_length',
                             truncation=True,
                             add_special_tokens=True,
                             return_attention_mask=True,
                             return_token_type_ids=True)
        return tokens

    tokenized = dataset.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing (num_proc={num_proc})",
    )

    if not wrap:
        return tokenized

    # Group texts for wrapped datasets
    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS, 
        insert_special_tokens=insert_special_tokens)
    
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

def _build_datasets(
    cfg: DictConfig,
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: Optional[int] = None,
):
    raw_dataset = load_raw_dataset(cfg)
    
    dataset_name = cfg.data.train if cfg.mode == "train" else cfg.data.valid
    
    # Determine wrap setting
    wrap = cfg.data.get('wrap', True)
    
    # Get insert_eos and insert_special_tokens settings
    insert_train_eos = cfg.data.get('insert_train_eos', True)
    insert_train_special = cfg.data.get('insert_train_special', True)
    insert_valid_eos = cfg.data.get('insert_valid_eos', True)
    insert_valid_special = cfg.data.get('insert_valid_special', True)

    # Special handling for datasets that come pre-split
    from datasets import DatasetDict
    if isinstance(raw_dataset, DatasetDict):
        # text8, some others come as DatasetDict
        train_raw = raw_dataset['train'] if 'train' in raw_dataset and not skip_train else None
        
        # Determine validation split name
        if dataset_name in ['text8', 'text8-crop', 'lm1b', 'ag_news']:
            valid_split = 'test'
        else:
            valid_split = 'validation'
        
        valid_raw = raw_dataset[valid_split] if valid_split in raw_dataset and not skip_valid else None
        
        # Apply max samples limits ΕΔΏ ΚΡΑΤΆΜΕ ΤΟΝ ΈΛΕΓΧΟ ΜΕΓΈΘΟΥΣ
        if train_raw is not None and cfg.data.get("max_train_samples"):
            max_samples = min(cfg.data.max_train_samples, len(train_raw))
            train_raw = train_raw.select(range(max_samples))
        
        if valid_raw is not None and cfg.data.get("max_valid_samples"):
            max_samples = min(cfg.data.max_valid_samples, len(valid_raw))
            valid_raw = valid_raw.select(range(max_samples))
        
        text_column = 'text'
        
    elif dataset_name in ['lambada', 'openwebtext-train', 'openwebtext-valid']:
        # These don't need splitting
        if dataset_name == 'lambada':
            train_raw = None
            valid_raw = raw_dataset
        else:
            train_raw = raw_dataset if not skip_train else None
            valid_raw = None
        
        # Apply max samples limits
        if train_raw is not None and cfg.data.get("max_train_samples"):
            max_samples = min(cfg.data.max_train_samples, len(train_raw))
            train_raw = train_raw.select(range(max_samples))
        
        if valid_raw is not None and cfg.data.get("max_valid_samples"):
            max_samples = min(cfg.data.max_valid_samples, len(valid_raw))
            valid_raw = valid_raw.select(range(max_samples))
        
        text_column = 'text'
    
    else:
        # Normal datasets - do manual split
        text_column = (
            cfg.data.text_column
            if "text_column" in cfg.data and cfg.data.text_column is not None
            else _default_text_column(raw_dataset)
        )

        # ΚΡΑΤΆΜΕ ΤΟΝ ΈΛΕΓΧΟ ΜΕΓΈΘΟΥΣ ΕΔΏ
        train_raw, valid_raw = manual_train_valid_split(
            raw_dataset,
            max_train_samples=cfg.data.get("max_train_samples"),
            max_valid_samples=cfg.data.get("max_valid_samples"),
            seed=valid_seed if valid_seed is not None else cfg.data.get("seed", 42),
        )
    
    # Handle special text columns for specific datasets
    if dataset_name == 'ptb':
        text_column = 'sentence'
    elif dataset_name.startswith('scientific_papers'):
        text_column = 'article'

    # Tokenization
    num_proc = cfg.data.get("num_proc", 4)
    block_size = cfg.model.length

    train_ds = None
    if not skip_train and train_raw is not None:
        train_ds = tokenize_dataset(
            train_raw,
            tokenizer,
            text_column,
            block_size,
            num_proc,
            dataset_name=dataset_name,
            wrap=wrap,
            insert_eos=insert_train_eos,
            insert_special_tokens=insert_train_special,
        )

    valid_ds = None
    if not skip_valid and valid_raw is not None:
        valid_ds = tokenize_dataset(
            valid_raw,
            tokenizer,
            text_column,
            block_size,
            num_proc,
            dataset_name=dataset_name,
            wrap=wrap,
            insert_eos=insert_valid_eos,
            insert_special_tokens=insert_valid_special,
        )

    return train_ds, valid_ds



# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

def collate_fn(batch):
    input_ids = torch.tensor(
        [x["input_ids"] for x in batch],
        dtype=torch.long
    )

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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
    cfg: DictConfig,
    tokenizer,
    skip_train: bool = False,
    skip_valid: bool = False,
    valid_seed: Optional[int] = None,
):
    train_ds, valid_ds = _build_datasets(
        cfg,
        tokenizer,
        skip_train=skip_train,
        skip_valid=skip_valid,
        valid_seed=valid_seed,
    )

    train_loader = None
    if train_ds is not None:
        train_loader = build_dataloader(
            train_ds,
            batch_size=cfg.loader.batch_size,
            shuffle=True,
            num_workers=cfg.loader.get("num_workers", 4),
        )
        # 🔑 REQUIRED BY main.py
        train_loader.tokenizer = tokenizer

    valid_loader = None
    if valid_ds is not None:
        valid_loader = build_dataloader(
            valid_ds,
            batch_size=cfg.loader.eval_batch_size,
            shuffle=False,
            num_workers=cfg.loader.get("num_workers", 4),
        )
        # 🔑 REQUIRED BY main.py
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


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop('shuffle', None)
        super().__init__(*args, generator=generator, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'random_state': self.generator.get_state(),
                'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get('random_state'))
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {'epoch': self.epoch, 'counter': self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.counter = state_dict['counter']
        self.restarting = True

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(
                    padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter:]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0