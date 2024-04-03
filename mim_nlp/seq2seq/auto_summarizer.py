from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import more_itertools
import numpy as np
import torch
from numpy._typing import _ArrayLikeStr_co
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError
from torch.nn import Module
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from mim_nlp.models import Summarizer
from mim_nlp.seq2seq.data import Seq2SeqDataCollator, Seq2SeqDataset


class AutoSummarizer(Summarizer):
    def __init__(
        self,
        max_length: int,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        neural_network: Optional[Module] = None,
        pretrained_model_name_or_path: Optional[str] = None,
        training_args: Optional[TrainingArguments] = None,
        generate_kwargs: Optional[dict[str, Any]] = None,
        device: str = "cuda:0",
    ) -> None:
        self._fitted = True
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if not neural_network:
            neural_network = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
            self._fitted = False
        self.max_length = max_length
        self.training_args = training_args
        self.trainer: Optional[Trainer] = None
        if generate_kwargs is None:
            generate_kwargs = {}
        self.generate_kwargs = generate_kwargs
        self._params = {"max_length": self.max_length}

        self.tokenizer = tokenizer
        self.neural_network = neural_network

        self.device = device

    def _fit(self, train_dataset: Seq2SeqDataset, eval_dataset: Optional[Seq2SeqDataset] = None) -> None:
        collator = Seq2SeqDataCollator(self.tokenizer, max_length=self.max_length)
        trainer = Trainer(
            model=self.neural_network,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            args=self.training_args,
        )
        trainer.train()
        self.trainer = trainer
        self._fitted = True

    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeStr_co) -> None:
        dataset = Seq2SeqDataset(x_train, y_train)
        self._fit(dataset)

    def fit_eval(
        self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeStr_co, x_eval: _ArrayLikeStr_co, y_eval: _ArrayLikeStr_co
    ) -> None:
        train_dataset = Seq2SeqDataset(x_train, y_train)
        eval_dataset = Seq2SeqDataset(x_eval, y_eval)
        self._fit(train_dataset, eval_dataset)

    def predict(self, x: _ArrayLikeStr_co, batch_size: Optional[int] = None, **generate_kwargs) -> NDArray[np.str_]:
        if not self._fitted:
            raise NotFittedError("Call .fit before trying to predict")
        if batch_size is None:
            batch_size = len(x)
        generate_kwargs = {**self.generate_kwargs, **generate_kwargs}
        batches = list(more_itertools.chunked(x, batch_size))
        outputs = list()
        for batch in tqdm(batches, desc=f"predicting in batches of size {batch_size}"):
            tokenized_output = self.tokenizer(batch, max_length=self.max_length, padding=True, truncation=True)
            input_ids = torch.tensor(tokenized_output["input_ids"]).to(self.device)
            attention_mask = torch.tensor(tokenized_output["attention_mask"]).to(self.device)
            generation_outputs = self.neural_network.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
            )
            outputs.extend(generation_outputs)
        y = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return np.array(y)

    def save(self, model_dir: Union[str, Path]) -> None:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as file:
            json.dump(self._params, file)
        self.tokenizer.save_pretrained(model_dir)
        torch.save(self.neural_network, model_dir / "model.bin")

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> AutoSummarizer:
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as file:
            params = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        neural_network = torch.load(f=model_dir / "model.bin")
        return cls(
            **params,
            tokenizer=tokenizer,
            neural_network=neural_network,
            pretrained_model_name_or_path=None,
        )
