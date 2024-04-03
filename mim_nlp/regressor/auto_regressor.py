from __future__ import annotations

import json
from os import environ
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional, Union

import numpy as np
import torch.nn as nn
from numpy._typing import NDArray, _ArrayLikeFloat_co, _ArrayLikeStr_co
from torch import Tensor
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from mim_nlp.models import Regressor


class DictDataset(Dataset):
    def __init__(self, tokens: BatchEncoding, labels: Optional[_ArrayLikeFloat_co] = None):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        result = {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx]}
        if self.labels is not None:
            result["labels"] = self.labels[idx]
        return result


class AutoRegressor(Regressor):
    """AutoRegressor is based on transformers' Auto Classes.

    The `input_size` parameter denotes the length of a tokenized text.
    A tokenized text is padded or truncated to a constant size equal to the `input_size`.
    Maximum `input_size` depends on the chosen model, the default is 512 tokens.

    The `device` parameter can have the following values:
        - `"cpu"` - The model will be loaded on the CPU.
        - `"cuda"` - The model will be loaded on a single GPU.
        - `"cuda:i"` - The model will be loaded on the specific GPU with the index `i`.

    It is also possible to use multiple GPUs. To do this:
        - Set `device` to `"cuda"`.
        - Set `many_gpus` to `True`.
        - As default, it will use all of them.

    To use only selected GPUs - set the environmental variable `CUDA_VISIBLE_DEVICES`.
    """

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        pretrained_model_name_or_path: str,
        input_size: int = 512,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self.device = device
        self.many_gpus = many_gpus

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=1, problem_type="regression"
        )
        self.loss_function = nn.MSELoss()

        device_split = self.device.split(":")
        no_cuda = True
        if device_split[0] == "cuda":
            no_cuda = False
            if not self.many_gpus:
                device_nr = device_split[1]
                environ["CUDA_VISIBLE_DEVICES"] = device_nr
        self.no_cuda = no_cuda

    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeFloat_co) -> None:
        train_dataset = DictDataset(self._tokenize(x_train), Tensor(y_train).float())
        self._fit(train_dataset)

    def fit_eval(
        self,
        x_train: _ArrayLikeStr_co,
        y_train: _ArrayLikeFloat_co,
        x_eval: _ArrayLikeStr_co,
        y_eval: _ArrayLikeFloat_co,
    ) -> None:
        train_dataset = DictDataset(self._tokenize(x_train), Tensor(y_train).float())
        eval_dataset = DictDataset(self._tokenize(x_eval), Tensor(y_eval).float())
        self._fit(train_dataset, eval_dataset)

    def _fit(self, train_dataset: DictDataset, eval_dataset: Optional[DictDataset] = None) -> None:
        def compute_mse_loss(eval_pred):
            predictions, labels = eval_pred
            return self.loss_function(predictions, labels)

        evaluation_strategy = "no" if eval_dataset is None else "epoch"
        with TemporaryDirectory() as tmp_path:
            training_args = TrainingArguments(
                output_dir=tmp_path,
                overwrite_output_dir=False,
                save_strategy="no",
                logging_strategy="epoch",
                evaluation_strategy=evaluation_strategy,
                optim="adamw_torch",
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                no_cuda=self.no_cuda,
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_mse_loss,
            )
            trainer.train()

    def predict(self, x: _ArrayLikeStr_co) -> NDArray[np.float64]:
        with TemporaryDirectory() as tmp_path:
            training_args = TrainingArguments(output_dir=tmp_path)
            trainer = Trainer(args=training_args, model=self.model)
            predictions = trainer.predict(DictDataset(self._tokenize(x))).predictions
        return predictions.flatten().astype(np.float64)

    def _tokenize(self, x: _ArrayLikeStr_co) -> BatchEncoding:
        return self.tokenizer(x, max_length=self.input_size, padding="max_length", truncation=True, return_tensors="pt")

    def save(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        params = self._get_model_params()
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as f:
            json.dump(params, f)

        self.tokenizer.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)

    def _get_model_params(self) -> dict[str, Any]:
        return {
            "model": self.__class__.__name__,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "input_size": self.input_size,
            "epochs": self.epochs,
            "device": self.device,
            "many_gpus": self.many_gpus,
        }

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: str = "cuda:0", many_gpus: bool = False) -> AutoRegressor:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as f:
            params = json.load(f)

        return cls(
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            pretrained_model_name_or_path=str(model_dir),
            input_size=params["input_size"],
            device=device,
            many_gpus=many_gpus,
        )
