from tf_2.segmentation.ssdd.utils.early_stop import EarlyStopping
import unittest
import tensorflow as tf


class TestEarlyStopping(unittest.TestCase):
    def test_validate_increase(self):
        patience = 3
        early = EarlyStopping(patience=patience)

        # 증가 시 Test
        losses = [1.333, 1.334, 1.335, 1.336, 1.337]
        last_idx = len(losses) - 1
        for idx in range(len(losses)):
            if idx != last_idx:
                self.assertFalse(early.validate_increase(losses[idx]))
            else:
                self.assertTrue(early.validate_increase(losses[idx]))

        # 증가 후 감소 후 증가 시 Test
        early = EarlyStopping(patience=patience)
        losses = [2.6, 3.7, 2.92, 2.95, 3.02, 3.24, 3.25]
        last_idx = len(losses) - 1
        for idx in range(len(losses)):
            if idx != last_idx:
                self.assertFalse(early.validate_increase(losses[idx]))
            else:
                self.assertTrue(early.validate_increase(losses[idx]))

    def test_validate_decrease(self):
        patience = 3

        early = EarlyStopping(patience=patience, decrease_target_grad=0.001)

        # 감소만 할때 Test
        losses = tf.constant([1.333, 1.323, 1.322, 1.322, 1.322, 1.322])
        last_idx = len(losses) - 1
        for idx in range(len(losses)):
            if idx != last_idx:
                self.assertFalse(early.validate_decrease(losses[idx]))
            else:
                self.assertTrue(early.validate_decrease(losses[idx]))

        early = EarlyStopping(patience=patience, decrease_target_grad=0.001)
        # 폭이 감소해 갈때 Test
        losses = tf.constant([0.018, 0.016, 0.017, 0.0165, 0.016, 0.015])
        last_idx = len(losses) - 1
        for idx in range(len(losses)):
            if idx != last_idx:
                self.assertFalse(early.validate_decrease(losses[idx]))
            else:
                self.assertTrue(early.validate_decrease(losses[idx]))
