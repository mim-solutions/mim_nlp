from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pickle import PicklingError
from typing import Callable, Iterable

import numpy as np
from gensim.utils import to_unicode
from numpy._typing import _ArrayLikeStr_co
from numpy.typing import NDArray

from mim_nlp.preprocessing import FunctionCannotBePickledException


class TextCleaner:
    def __init__(self, pipeline: Iterable[Callable[[str], str]]) -> None:
        self.pipeline = pipeline

    def clean_texts(
        self, texts: _ArrayLikeStr_co, multiprocessing: bool = True, cpu_workers=cpu_count()
    ) -> NDArray[np.str_]:
        if multiprocessing:
            clean_texts = self._clean_texts_with_multiprocessing(texts, cpu_workers)
        else:
            clean_texts = [self._clean_text(txt) for txt in texts]
        return np.array(clean_texts)

    def _clean_texts_with_multiprocessing(self, texts: _ArrayLikeStr_co, cpu_workers=cpu_count()) -> NDArray[np.str_]:
        try:
            p = Pool(cpu_workers)
            clean_texts = p.map(self._clean_text, texts)
            return np.array(clean_texts)
        except (AttributeError, PicklingError):
            raise FunctionCannotBePickledException()

    def _clean_text(self, txt: str) -> str:
        txt = to_unicode(txt)
        for f in self.pipeline:
            txt = f(txt)
        return txt
