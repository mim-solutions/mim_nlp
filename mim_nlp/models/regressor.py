from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeStr_co
from numpy.typing import NDArray

from mim_nlp.models.model import Model


class Regressor(Model, ABC):
    @abstractmethod
    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeFloat_co) -> None:
        pass

    @abstractmethod
    def predict(self, x: _ArrayLikeStr_co) -> NDArray[np.float64]:
        pass
