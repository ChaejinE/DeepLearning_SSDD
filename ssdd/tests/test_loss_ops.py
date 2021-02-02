from unittest import TestCase
from tf_2.segmentation.ssdd.train.loss_ops import classification_loss

import tensorflow as tf


class TestLossOps(TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownclass(cls):
        pass

    def test_classification_loss(self):
        # TEST 1. 정상 동작하고 계산 값도 틀리지 않는지 확인
        tp_loss = classification_loss(y_true=tf.constant([1]),
                                      y_pred=tf.constant([[10., 100000000.]]))

        tn_loss = classification_loss(y_true=tf.constant([0]),
                                      y_pred=tf.constant([[1000., 10.]]))

        fp_loss = classification_loss(y_true=tf.constant([1]),
                                      y_pred=tf.constant([[10000., 10.]]))

        fn_loss = classification_loss(y_true=tf.constant([0]),
                                      y_pred=tf.constant([[100., 100000.]]))

        self.assertTrue(tp_loss < fp_loss)
        self.assertTrue(tp_loss < fn_loss)
        self.assertTrue(tn_loss < fp_loss)
        self.assertTrue(tn_loss < fn_loss)

        # TEST 2. y_true 의 type 은 int, float 둘 다 가능

        loss = classification_loss(y_true=tf.constant([1.]),
                                   y_pred=tf.constant([[10., 100000000.]]))

        self.assertIsNotNone(loss)

        # TEST 3. y_true 의 shape 은 (batch size) 또는 (batch size, 1) 둘 다 가능

        loss = classification_loss(y_true=tf.constant([[1.], [1.]]),
                                   y_pred=tf.constant([[10., 100000000.], [10., 100000000.]]))

        self.assertIsNotNone(loss)

        # TEST 4. y_pred 는 반드시 float 값 이어야 함

        self.assertRaises(Exception,
                          classification_loss,
                          y_true=tf.constant([1.]),
                          y_pred=tf.constant([[10, 100000000]]))

        # TEST 5. y_pred 는 shape 이 (batch size, num class) 이어야 함. (이 때 num class 는 y_true 값 중 max 값에 +1 한 값이 됨.)

        self.assertRaises(Exception,
                          classification_loss,
                          y_true=tf.constant([1.]),
                          y_pred=tf.constant([[10.]]))
