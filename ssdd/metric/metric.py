from typing import *


class Metric:
    def __init__(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    @property
    def tp(self):

        return self._tp

    @property
    def fp(self):

        return self._fp

    @property
    def tn(self):

        return self._tn

    @property
    def fn(self):

        return self._fn

    @property
    def precision(self):

        return self.tp / max((self.tp + self.fp), 1)

    @property
    def recall(self):

        return self.tp / max((self.tp + self.fn), 1)

    @property
    def accuracy(self):

        return (self.tp + self.tn) / max((self.tp + self.tn + self.fp + self.fn), 1)

    @property
    def f1_score(self):

        return 2 * ((self.precision * self.recall) / max((self.precision + self.recall), 1))

    @tp.setter
    def tp(self, value):
        self._tp = value

    @fp.setter
    def fp(self, value):
        self._fp = value

    @tn.setter
    def tn(self, value):
        self._tn = value

    @fn.setter
    def fn(self, value):
        self._fn = value

    def update(self, tp_cnt=None, fp_cnt=None, tn_cnt=None, fn_cnt=None):
        if tp_cnt is not None:
            self.tp += tp_cnt

        if fp_cnt is not None:
            self.fp += fp_cnt

        if tn_cnt is not None:
            self.tn += tn_cnt

        if fn_cnt is not None:
            self.fn += fn_cnt

    def classify_and_update(self, y_true: bool, y_pred: bool) -> str:
        if y_true:
            if y_pred:
                self.tp += 1

                return 'tp'
            else:
                self.fn += 1

                return 'fn'
        else:
            if y_pred:
                self.fp += 1

                return 'fp'
            else:
                self.tn += 1

                return 'tn'

    def clear(self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
