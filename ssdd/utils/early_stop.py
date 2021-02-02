import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 0, decrease_target_grad: float = 0):
        self._inc_step: int = 0
        self._dec_step: int = 0
        self._inc_loss: float = 10.
        self._dec_loss: float = 10.
        self._grad: float = 10.
        self._decrease_target_grad: float = decrease_target_grad
        self.patience: int = patience

    def validate_increase(self, loss: float):
        if self._inc_loss < loss:
            self._inc_step += 1
            self._inc_loss = loss
            if self._inc_step > self.patience:

                return True
        else:
            self._inc_step = 0
            self._inc_loss = loss

        return False

    def validate_decrease(self, loss: float, step: int = 1):
        cur_grad = abs((self._dec_loss - loss) / step)
        self._dec_loss = loss
        if cur_grad < self._decrease_target_grad or np.isclose(cur_grad, self._decrease_target_grad):
            self._dec_step += 1
            if self._dec_step > self.patience:

                return True
        else:
            self._dec_step = 0

        self._grad = cur_grad

        return False

    def check_state(self, loss):

        return any([self.validate_increase(loss), self.validate_decrease(loss)])
