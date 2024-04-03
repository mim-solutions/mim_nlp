from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

from numpy._typing import _ArrayLikeStr_co
from numpy.typing import NDArray


class Model(ABC):
    @abstractmethod
    def fit(self, x_train: _ArrayLikeStr_co) -> None:
        pass

    @abstractmethod
    def predict(self, x: _ArrayLikeStr_co) -> NDArray[Any]:
        pass

    @abstractmethod
    def save(self, model_dir: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Union[str, Path]) -> Model:
        pass
