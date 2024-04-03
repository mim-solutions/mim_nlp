from abc import ABC, abstractmethod

import numpy as np
from numpy._typing import _ArrayLikeInt_co, _ArrayLikeStr_co
from numpy.typing import NDArray

from mim_nlp.models.model import Model


class Classifier(Model, ABC):
    @abstractmethod
    def fit(self, x_train: _ArrayLikeStr_co, y_train: _ArrayLikeInt_co) -> None:
        pass

    @abstractmethod
    def predict(self, x: _ArrayLikeStr_co) -> NDArray[np.int64]:
        pass

    @abstractmethod
    def predict_scores(self, x: _ArrayLikeStr_co) -> NDArray[np.float64]:
        pass
