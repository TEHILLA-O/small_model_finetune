"""Dataset loading. Every split is built from fixed public datasets with a fixed seed."""
from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class Split:
    name: str
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)


def _sample(texts, labels, n, seed):
    idx = list(range(len(labels)))
    if n is not None and n < len(idx):
        rng = random.Random(seed)
        rng.shuffle(idx)
        idx = sorted(idx[:n])
    return [texts[i] for i in idx], [int(labels[i]) for i in idx]


def sst2_train_dev(train_size: int, dev_size: int, data_seed: int) -> tuple[Split, Split]:
    """Train and dev both come from the SST-2 train split and never overlap.

    The official SST-2 validation set is kept untouched for final evaluation.
    """
    ds = load_dataset("stanfordnlp/sst2", split="train")
    texts, labels = ds["sentence"], ds["label"]
    idx = list(range(len(labels)))
    random.Random(data_seed).shuffle(idx)
    dev_idx = sorted(idx[:dev_size])
    train_idx = sorted(idx[dev_size : dev_size + train_size])
    dev = Split("sst2_dev", [texts[i] for i in dev_idx], [int(labels[i]) for i in dev_idx])
    train = Split("sst2_train", [texts[i] for i in train_idx], [int(labels[i]) for i in train_idx])
    return train, dev


def eval_sets(imdb_size: int = 2000, seed: int = 0) -> list[Split]:
    """In-domain plus two out-of-domain sentiment test sets."""
    sst = load_dataset("stanfordnlp/sst2", split="validation")
    rt = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    imdb = load_dataset("stanfordnlp/imdb", split="test")
    it, il = _sample(imdb["text"], imdb["label"], imdb_size, seed)
    return [
        Split("sst2_validation", list(sst["sentence"]), [int(x) for x in sst["label"]]),
        Split("rotten_tomatoes_test", list(rt["text"]), [int(x) for x in rt["label"]]),
        Split(f"imdb_test_{imdb_size}", it, il),
    ]
