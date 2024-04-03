from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import _ArrayLikeStr_co
from numpy.typing import NDArray

from mim_nlp.models.model import Model


class Seq2Seq(Model, ABC):
    @abstractmethod
    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeStr_co) -> None:
        pass

    @abstractmethod
    def predict(self, x: _ArrayLikeStr_co) -> NDArray[np.str_]:
        pass
