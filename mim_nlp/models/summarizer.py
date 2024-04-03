from abc import ABC

from mim_nlp.models.seq2seq import Seq2Seq


class Summarizer(Seq2Seq, ABC):
    pass
