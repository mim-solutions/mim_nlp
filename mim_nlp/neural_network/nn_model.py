from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt_co, _ArrayLikeStr_co
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from mim_nlp.neural_network.nn_module import NNModule


class NNModelMixin:
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        input_size: int,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, Pipeline, TfidfVectorizer]],
        neural_network: nn.Module,
        loss_function: Union[_Loss, Callable[[Any, Any], Any]],
        optimizer: type[Optimizer] = Adam,
        optimizer_params: Optional[dict[str, Any]] = None,
        train_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
        eval_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
        callbacks: Optional[Union[Callback, list[Callback]]] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_size = input_size
        self.callbacks = callbacks
        self.device = device
        self.many_gpus = many_gpus
        self.is_fitted = False

        if not isinstance(tokenizer, (PreTrainedTokenizerBase, Pipeline, TfidfVectorizer)):
            raise TypeError(
                "This tokenizer is not supported! Check `__init__` type hint for the list of supported classes!"
            )
        self.tokenizer = tokenizer
        self.nn_module = NNModule(
            neural_network=neural_network,
            loss_function=loss_function,
            optimizer_class=optimizer,
            optimizer_params=optimizer_params,
            train_metrics_dict=train_metrics_dict,
            eval_metrics_dict=eval_metrics_dict,
        )

        device_split = device.split(":")
        device_nr = "auto"
        accelerator = "cpu"
        if device_split[0] == "cuda":
            accelerator = "gpu"
            device_nr = [int(device_split[1])]
        if many_gpus:
            device_nr = -1

        self.accelerator = accelerator
        self.device_nr = device_nr

        self.trainer = Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=device_nr,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=False,
        )

    def _fit(
        self,
        x_train: _ArrayLikeStr_co,
        y_train: Tensor,
        x_eval: Optional[_ArrayLikeStr_co] = None,
        y_eval: Optional[Tensor] = None,
        fit_tokenizer: bool = False,
        is_classification: bool = False,
        is_multiclass: bool = False,
    ) -> None:
        self.nn_module.is_classification = is_classification
        self.nn_module.is_multiclass = is_multiclass
        self.is_fitted = True

        if self.epochs > self.trainer.max_epochs:
            new_trainer = Trainer(
                max_epochs=self.epochs,
                accelerator=self.accelerator,
                devices=self.device_nr,
                callbacks=self.callbacks,
                enable_checkpointing=False,
                logger=False,
            )

            if not self.nn_module.optimizer_state:
                try:
                    # save optimizer state to restore after changing trainer object
                    self.nn_module.optimizer_state = self.nn_module.optimizers().optimizer.state_dict()
                except RuntimeError as e:
                    if "is not attached to a `Trainer`" not in str(e):
                        raise e

            self.trainer = new_trainer

        if fit_tokenizer:
            self.fit_tokenizer(x_train, y_train)

        x_train_tokenized = self._tokenize(x_train)
        train_loader = DataLoader(TensorDataset(x_train_tokenized, y_train), batch_size=self.batch_size)

        eval_loader = None
        if x_eval is not None:
            x_eval_tokenized = self._tokenize(x_eval)
            eval_loader = DataLoader(TensorDataset(x_eval_tokenized, y_eval), batch_size=self.batch_size)

        self.trainer.fit(self.nn_module, train_loader, eval_loader)

    def fit_tokenizer(
        self, x_train: _ArrayLikeStr_co, y_train: Optional[Union[_ArrayLikeInt_co, _ArrayLikeFloat_co]] = None
    ) -> None:
        if not isinstance(self.tokenizer, (Pipeline, TfidfVectorizer)):
            raise TypeError("This method supports only scikit-learn's classes (`Pipeline` or `TfidfVectorizer`).")
        self.tokenizer.fit(x_train, y_train)

    def _test(
        self,
        x: _ArrayLikeStr_co,
        y_test: Tensor,
        batch_size: Optional[int] = None,
        test_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
    ) -> dict[str, Any]:
        if not self.is_fitted:
            raise NotFittedError("Call .fit before trying to predict")

        if batch_size is None:
            batch_size = self.batch_size

        if test_metrics_dict is not None:
            for k, v in test_metrics_dict.items():
                if isinstance(v, nn.Module):
                    self.nn_module.test_metrics_module_dict[k] = v
                else:
                    self.nn_module.test_metrics_dict[k] = v
        else:
            self.nn_module.test_metrics_module_dict = nn.ModuleDict()
            self.nn_module.test_metrics_dict = {}

        x_tokenized = self._tokenize(x)
        test_loader = DataLoader(TensorDataset(x_tokenized, y_test), batch_size=batch_size)
        return self.trainer.test(self.nn_module, test_loader)[0]

    def _get_predictions(self, x: _ArrayLikeStr_co, batch_size: Optional[int] = None) -> Tensor:
        if not self.is_fitted:
            raise NotFittedError("Call .fit before trying to predict")
        if batch_size is None:
            batch_size = self.batch_size
        x_tokenized = self._tokenize(x)
        test_loader = DataLoader(TensorDataset(x_tokenized), batch_size=batch_size)
        predictions = self.trainer.predict(self.nn_module, test_loader)
        predictions = torch.cat(predictions)
        return self.nn_module.convert_logits_to_probabilities(predictions)

    def _tokenize(self, x: _ArrayLikeStr_co) -> Tensor:
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            return self.tokenizer(
                x, max_length=self.input_size, padding="max_length", truncation=True, return_tensors="pt"
            )["input_ids"].float()
        elif isinstance(self.tokenizer, (Pipeline, TfidfVectorizer)):
            return Tensor(self.tokenizer.transform(x).toarray()).float()
        else:
            raise TypeError(
                "This tokenizer is not supported! Check `__init__` type hint for the list of supported classes!"
            )

    def save(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        params = self._get_model_params()
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as f:
            json.dump(params, f)

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self.tokenizer.save_pretrained(model_dir)
        else:
            with open(model_dir / "tokenizer.pickle", mode="wb") as f:
                pickle.dump(self.tokenizer, f)

        torch.save(self.nn_module.neural_network, model_dir / "neural_network.bin")
        torch.save(self.nn_module.loss_fun, model_dir / "loss_function.bin")
        torch.save(self.nn_module.optimizer_class, model_dir / "optimizer_class.bin")
        try:
            torch.save(self.nn_module.optimizers().optimizer.state_dict(), model_dir / "optimizer_state.bin")
        except RuntimeError as e:
            if "is not attached to a `Trainer`" not in str(e):
                raise e
        torch.save(self.nn_module.train_metrics_module_dict, model_dir / "train_metrics_module_dict.bin")
        torch.save(self.nn_module.eval_metrics_module_dict, model_dir / "eval_metrics_module_dict.bin")
        torch.save(self.callbacks, model_dir / "callbacks.bin")

        with open(model_dir / "train_metrics_dict.pickle", mode="wb") as f:
            pickle.dump(self.nn_module.train_metrics_dict, f)
        with open(model_dir / "eval_metrics_dict.pickle", mode="wb") as f:
            pickle.dump(self.nn_module.eval_metrics_dict, f)

    def save_slim(self, model_dir: Union[str, Path]) -> None:
        """Saves model for inference only, which reduces required memory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        params = self._get_model_params()
        params.pop("optimizer_params")
        with open(file=model_dir / "params.json", mode="w", encoding="utf-8") as f:
            json.dump(params, f)

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            self.tokenizer.save_pretrained(model_dir)
        else:
            self.drop_stop_words_set()
            with open(model_dir / "tokenizer.pickle", mode="wb") as f:
                pickle.dump(self.tokenizer, f)

        torch.save(self.nn_module.neural_network, model_dir / "neural_network.bin")

    def save_without_stop_words(self, model_dir: Union[str, Path]) -> None:
        if not isinstance(self.tokenizer, (Pipeline, TfidfVectorizer)):
            raise TypeError("This method supports only scikit-learn's classes (`Pipeline` or `TfidfVectorizer`).")
        self.drop_stop_words_set()
        self.save(model_dir)

    def drop_stop_words_set(self) -> None:
        """The method deletes the set of stopwords saved in the tf-idf part of the model.

        For more info see the documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.
        """
        if not isinstance(self.tokenizer, (Pipeline, TfidfVectorizer)):
            raise TypeError("This method supports only scikit-learn's classes (`Pipeline` or `TfidfVectorizer`).")

        if isinstance(self.tokenizer, TfidfVectorizer):
            self.tokenizer.stop_words_ = None
        else:
            for i in range(len(self.tokenizer.steps)):
                if isinstance(self.tokenizer[i], TfidfVectorizer):
                    self.tokenizer[i].stop_words_ = None

    def _get_model_params(self) -> dict[str, Any]:
        pre_trained = False
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            pre_trained = True
        return {
            "model": self.__class__.__name__,
            "pre_trained_tokenizer": pre_trained,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "input_size": self.input_size,
            "optimizer_params": self.nn_module.optimizer_params,
            "device": self.device,
            "many_gpus": self.many_gpus,
            "is_fitted": self.is_fitted,
            "is_classification": self.nn_module.is_classification,
            "is_multiclass": self.nn_module.is_multiclass,
        }

    @classmethod
    def load(cls, model_dir: Union[str, Path], device: str = "cuda:0", many_gpus: bool = False) -> NNModelMixin:
        model_dir = Path(model_dir)
        with open(file=model_dir / "params.json", mode="r", encoding="utf-8") as f:
            params = json.load(f)

        if params["pre_trained_tokenizer"]:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            with open(model_dir / "tokenizer.pickle", mode="rb") as f:
                tokenizer = pickle.load(f)

        neural_network = torch.load(model_dir / "neural_network.bin")
        loss_function_path = model_dir / "loss_function.bin"
        loss_function = torch.load(loss_function_path) if loss_function_path.exists() else None
        optimizer_class_path = model_dir / "optimizer_class.bin"
        optimizer_class = torch.load(optimizer_class_path) if optimizer_class_path.exists() else None
        callbacks_path = model_dir / "callbacks.bin"
        callbacks = torch.load(callbacks_path) if callbacks_path.exists() else None
        train_metrics_mod_path = model_dir / "train_metrics_module_dict.bin"
        train_metrics_module_dict = torch.load(train_metrics_mod_path) if train_metrics_mod_path.exists() else None
        eval_metrics_mod_path = model_dir / "eval_metrics_module_dict.bin"
        eval_metrics_module_dict = torch.load(eval_metrics_mod_path) if eval_metrics_mod_path.exists() else None
        optimizer_state_path = model_dir / "optimizer_state.bin"
        optimizer_state = torch.load(optimizer_state_path) if optimizer_state_path.exists() else None

        train_metrics_path = model_dir / "train_metrics_dict.pickle"
        train_metrics_dict = {}
        if train_metrics_path.exists():
            with open(train_metrics_path, mode="rb") as f:
                train_metrics_dict = pickle.load(f)

        eval_metrics_path = model_dir / "eval_metrics_dict.pickle"
        eval_metrics_dict = {}
        if eval_metrics_path.exists():
            with open(eval_metrics_path, mode="rb") as f:
                eval_metrics_dict = pickle.load(f)

        model = cls(
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            input_size=params["input_size"],
            tokenizer=tokenizer,
            neural_network=neural_network,
            loss_function=loss_function,
            optimizer=optimizer_class,
            optimizer_params=params["optimizer_params"] if "optimizer_params" in params else None,
            callbacks=callbacks,
            device=device,
            many_gpus=many_gpus,
        )
        model.is_fitted = params["is_fitted"]
        model.nn_module.is_classification = params["is_classification"]
        model.nn_module.is_multiclass = params["is_multiclass"]
        if train_metrics_module_dict:
            model.nn_module.train_metrics_module_dict = train_metrics_module_dict
        if eval_metrics_module_dict:
            model.nn_module.eval_metrics_module_dict = eval_metrics_module_dict
        model.nn_module.train_metrics_dict = train_metrics_dict
        model.nn_module.eval_metrics_dict = eval_metrics_dict
        model.nn_module.optimizer_state = optimizer_state
        return model
