from itertools import chain
from typing import Any, Callable, Optional, Union

import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torchmetrics import Metric


class NNModule(LightningModule):
    def __init__(
        self,
        neural_network: nn.Module,
        loss_function: Union[_Loss, Callable[[Any, Any], Any]],
        train_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]],
        eval_metrics_dict: Optional[dict[str, Union[Metric, Callable[[Tensor, Tensor], Any]]]],
        optimizer_class: type[Optimizer],
        optimizer_params: Optional[dict[str, Any]],
    ):
        super().__init__()

        self.neural_network = neural_network
        self.loss_fun = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params if optimizer_params else {}
        self.optimizer_state: Optional[dict[str, Any]] = None
        self.is_classification = False
        self.is_multiclass = False

        # TorchMetrics' metrics contain internal states and cannot be used for both training and evaluation.
        # https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#common-pitfalls
        # `Module`s must be kept in `ModuleDict` to avoid the error concerning using different devices.
        self.train_metrics_module_dict = nn.ModuleDict()
        self.eval_metrics_module_dict = nn.ModuleDict()
        self.test_metrics_module_dict = nn.ModuleDict()
        self.train_metrics_dict = {}
        self.eval_metrics_dict = {}
        self.test_metrics_dict = {}

        if train_metrics_dict is not None:
            for k, v in train_metrics_dict.items():
                if isinstance(v, nn.Module):
                    self.train_metrics_module_dict[k] = v
                else:
                    self.train_metrics_dict[k] = v

        if eval_metrics_dict is not None:
            for k, v in eval_metrics_dict.items():
                if isinstance(v, nn.Module):
                    self.eval_metrics_module_dict[k] = v
                else:
                    self.eval_metrics_dict[k] = v

    def forward(self, x: Any) -> Any:
        return self.neural_network(x)

    def configure_optimizers(self) -> Optimizer:
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        if self.optimizer_state:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        return optimizer

    def convert_logits_to_probabilities(self, y: Tensor) -> Tensor:
        if not self.is_classification:
            return y
        if self.is_multiclass:
            return nn.Softmax()(y)
        return nn.Sigmoid()(y)

    def training_step(self, train_batch: Any, batch_idx: int) -> Any:
        x, y = train_batch
        y = y.squeeze()
        y_pred = self.forward(x).squeeze()

        loss = self.loss_fun(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name, metric in chain(self.train_metrics_module_dict.items(), self.train_metrics_dict.items()):
            y_probabilities = self.convert_logits_to_probabilities(y_pred)
            if isinstance(metric, Metric):
                metric(y_probabilities, y)
                self.log("train_" + name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                self.log(
                    "train_" + name,
                    metric(y_probabilities, y),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        return loss

    def validation_step(self, eval_batch: Any, batch_idx: int) -> Any:
        x, y = eval_batch
        y = y.squeeze()
        y_pred = self.forward(x).squeeze()

        loss = self.loss_fun(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        for name, metric in chain(self.eval_metrics_module_dict.items(), self.eval_metrics_dict.items()):
            y_probabilities = self.convert_logits_to_probabilities(y_pred)
            if isinstance(metric, Metric):
                metric(y_probabilities, y)
                self.log("val_" + name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            else:
                self.log(
                    "val_" + name, metric(y_probabilities, y), on_step=False, on_epoch=True, prog_bar=True, logger=True
                )

    def test_step(self, eval_batch: Any, batch_idx: int) -> dict[str, Any]:
        x, y = eval_batch
        y = y.squeeze()
        y_pred = self.forward(x).squeeze()
        y_probabilities = self.convert_logits_to_probabilities(y_pred)

        results = {}
        if len(self.test_metrics_module_dict) + len(self.test_metrics_dict) > 0:
            metrics = chain(self.test_metrics_module_dict.items(), self.test_metrics_dict.items())
        else:
            metrics = chain(self.eval_metrics_module_dict.items(), self.eval_metrics_dict.items())

        for name, metric in metrics:
            if isinstance(metric, Metric):
                metric(y_probabilities, y)
                self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                results["name"] = metric
            else:
                score = metric(y_probabilities, y)
                self.log(name, score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                results["name"] = score

        return results

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch = batch[0]  # fixes: DataLoader with TensorDataset wraps single Tensor in list
        return super().predict_step(batch, batch_idx, dataloader_idx)
