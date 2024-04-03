from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np
import torch.nn as nn
from numpy._typing import NDArray, _ArrayLikeInt_co, _ArrayLikeStr_co
from pytorch_lightning.callbacks import Callback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Adam, Optimizer
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase

from mim_nlp.models import Classifier
from mim_nlp.neural_network import NNModelMixin


class NNClassifier(NNModelMixin, Classifier):
    """Neural Network Classifier

    The `input_size` parameter denotes the length of a tokenized text.
    This should be equal to the size of the input layer in the neural network.
    In the case of using TF-IDF, the output size is constant and equal to the size of the vocabulary,
    so the `input_size` has to be set accordingly.
    When transformers' tokenizer is used,
    a tokenized text is padded or truncated to a constant size equal to the `input_size`.

    Neural network should omit activation function and return logits.
    Take that into consideration when choosing the loss function!
    We use Sigmoid / Softmax internally to get predictions.

    The `loss_function` is by default set to BCEWithLogitsLoss,
    which combines a Sigmoid layer and the BCELoss in one single class.
    For multiclass classification, use Cross Entropy Loss. Both losses take logits, as stated above.

    Callables in `metrics_dict` take predictions (as probabilities) and targets, in that order! Callables can't be
    lambda functions because they are not pickleable and it would cause problems with saving the model.

    Tips:
        - Change every lambda function to a function.
        - Set every argument in the function via `functools.partial`.

    Example:
        >>> def accuracy_binary(y_pred, y_target):
        >>>     y_pred = y_pred > 0.5
        >>>     return torch.sum(y_target == y_pred) / len(y_target)

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
        epochs: int,
        input_size: int,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, Pipeline, TfidfVectorizer]],
        neural_network: nn.Module,
        loss_function: Union[_Loss, Callable[[Any, Any], Any]] = nn.BCEWithLogitsLoss(),
        optimizer: type[Optimizer] = Adam,
        optimizer_params: Optional[dict[str, Any]] = None,
        train_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
        eval_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
        callbacks: Optional[Union[Callback, list[Callback]]] = None,
        device: str = "cuda:0",
        many_gpus: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            epochs=epochs,
            input_size=input_size,
            tokenizer=tokenizer,
            neural_network=neural_network,
            loss_function=loss_function,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            train_metrics_dict=train_metrics_dict,
            eval_metrics_dict=eval_metrics_dict,
            callbacks=callbacks,
            device=device,
            many_gpus=many_gpus,
        )

    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeInt_co, fit_tokenizer: bool = False) -> None:
        """For multiclass classifications `y_train` labels should be encoded as categorical, i.e. integers."""
        is_multiclass = False
        # check if multiclass
        if any(y >= 2 for y in y_train):
            y_train = Tensor(y_train).long()
            is_multiclass = True
        else:
            y_train = Tensor(y_train).float()
        super()._fit(
            x_train,
            y_train,
            x_eval=None,
            y_eval=None,
            fit_tokenizer=fit_tokenizer,
            is_classification=True,
            is_multiclass=is_multiclass,
        )

    def fit_eval(
        self,
        x_train: _ArrayLikeStr_co,
        y_train: _ArrayLikeInt_co,
        x_eval: _ArrayLikeStr_co,
        y_eval: _ArrayLikeInt_co,
        fit_tokenizer: bool = False,
    ) -> None:
        """For multiclass classifications `y` labels should be encoded as categorical, i.e. integers."""
        is_multiclass = False
        # check if multiclass
        if any(y >= 2 for y in y_train):
            y_train = Tensor(y_train).long()
            y_eval = Tensor(y_eval).long()
            is_multiclass = True
        else:
            y_train = Tensor(y_train).float()
            y_eval = Tensor(y_eval).float()
        super()._fit(
            x_train,
            y_train,
            x_eval,
            y_eval,
            fit_tokenizer=fit_tokenizer,
            is_classification=True,
            is_multiclass=is_multiclass,
        )

    def fit_tokenizer(self, x_train: _ArrayLikeStr_co, y_train: Optional[_ArrayLikeInt_co] = None) -> None:
        super().fit_tokenizer(x_train, y_train)

    def predict(
        self, x: _ArrayLikeStr_co, batch_size: Optional[int] = None, score_threshold: float = 0.5
    ) -> NDArray[np.int64]:
        predictions = self._get_predictions(x, batch_size)
        if predictions.shape[1] > 1:
            # multiclass classification
            return np.array(np.argmax(predictions, axis=1), dtype=np.int64)
        return np.array(predictions.flatten() > score_threshold, dtype=np.int64)

    def predict_scores(self, x: _ArrayLikeStr_co, batch_size: Optional[int] = None) -> NDArray[np.float64]:
        predictions = self._get_predictions(x, batch_size)
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
        return np.array(predictions, dtype=np.float64)

    def test(
        self,
        x: _ArrayLikeStr_co,
        y_test: _ArrayLikeInt_co,
        batch_size: Optional[int] = None,
        test_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]] = None,
    ) -> dict[str, Any]:
        if self.nn_module.is_multiclass:
            y_test = Tensor(y_test).long()
        else:
            y_test = Tensor(y_test).float()
        return super()._test(x, y_test, batch_size, test_metrics_dict)
