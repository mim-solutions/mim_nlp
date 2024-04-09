""" Everything that relates to processing the data from a raw DataFrame to the input to be fed to the model.
That includes:
* Seq2SeqPytorchDataset that fetches samples from a DataFrame
* Seq2SeqDataCollator that converts the source and target texts to tensors in batches
* performing train-test splits

Everything in this module can be used in general for sequence to sequence tasks, not only summarization.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from loguru import logger
from numpy._typing import _ArrayLikeStr_co
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class Seq2SeqModelInput:
    source: str
    target: str


class Seq2SeqDataset(Dataset):
    """Wraps a list of source (x_train) and target (y_train) strings into a pytorch Dataset."""

    def __init__(
        self,
        x_train: _ArrayLikeStr_co,
        y_train: _ArrayLikeStr_co,
        source_transform_fn: Optional[Callable[[str], str]] = None,
        target_transform_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        assert len(x_train) == len(y_train)
        self.x_train = x_train
        self.y_train = y_train
        if source_transform_fn is not None:
            self.x_train = [source_transform_fn(x) for x in self.x_train]
            logger.info(f"Transformed x_train with {source_transform_fn.__name__} function")
        if target_transform_fn is not None:
            self.y_train = [target_transform_fn(y) for y in self.y_train]
            logger.info(f"Transformed y_train with {target_transform_fn.__name__} function")

    def __getitem__(self, i) -> Seq2SeqModelInput:
        return Seq2SeqModelInput(source=self.x_train[i], target=self.y_train[i])

    def __len__(self):
        return len(self.x_train)


class Seq2SeqDataCollator:
    """Converts a list of samples from a pytorch Dataset (source-target string pairs) to batches of tensors to be
    fed into the model."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: Optional[int] = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[Seq2SeqModelInput]) -> dict[str, torch.Tensor]:
        inputs = [example.source for example in examples]
        summaries = [example.target for example in examples]
        input_encoding = self.tokenizer(inputs, padding=True, truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(input_encoding.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(input_encoding.attention_mask, dtype=torch.long)
        summaries_encoding = self.tokenizer(summaries, padding=True, truncation=True, max_length=self.max_length)
        labels = torch.tensor(summaries_encoding.input_ids, dtype=torch.long)
        labels = torch.masked_fill(
            labels, labels == self.tokenizer.pad_token_id, -100
        )  # -100 makes a CrossEntropyLoss ignore the given position
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
