from unittest import TestCase

import numpy as np
import tensorflow as tf
from tf_2.segmentation.ssdd.model.u_net import *


class TestUnet(TestCase):
    def test_conv_block(self):
        # TEST 1. 정상 동작
        batch = 3
        ch = 3
        size = 284
        diff = 4
        num_filter = 64

        inputs = np.ones((batch, size, size, ch))
        x = conv_block(inputs, num_filter=num_filter)

        self.assertListEqual([batch, size-diff, size-diff, num_filter], list(x.shape))

        # TEST 2. batch shape 이 없으면?
        inputs = np.ones((size, size, ch))
        self.assertRaises(ValueError,
                          conv_block,
                          inputs,
                          num_filter)

        # TEST 3. size 가 너무 작아서 conv 를 더 이상 할 수 없다면?
        inputs = np.ones((batch, 4, 4, ch))
        self.assertRaises(AssertionError,
                          conv_block,
                          inputs,
                          num_filter)

        # TEST 4. ch shape 이 없으면?
        inputs = np.ones((batch, size, size))
        self.assertRaises(ValueError,
                          conv_block,
                          inputs,
                          num_filter)

        # TEST 5. num filter 가 0 이면?
        inputs = np.ones((batch, size, size, ch))
        self.assertRaises(AssertionError,
                          conv_block,
                          inputs,
                          num_filter=0)

    def test_skip_connection(self):
        # TEST 1. 정상 동작
        x1 = np.zeros((1, 100, 100, 2))
        x2 = np.ones((1, 90, 90, 2))
        x = skip_connection(x1, x2)
        self.assertListEqual([1, 90, 90,  4], list(x.shape))
        self.assertListEqual([0, 0, 1, 1], list(x[0, 1, 1, :]))

        # TEST 2. center crop 이 안되는 크기인 경우
        x1 = np.zeros((1, 100, 100, 2))
        x2 = np.ones((1, 91, 91, 2))
        x = skip_connection(x1, x2)
        self.assertListEqual([1, 91, 91,  4], list(x.shape))

        # TEST 3. x1 < x2
        self.assertRaises(AssertionError,
                          skip_connection,
                          x1=x2,
                          x2=x1)

    def test_skip_connector(self):
        x1 = np.zeros((1, 100, 100, 2))
        x2 = np.ones((1, 90, 90, 2))
        conn = skip_connector(x1)

        x = conn(x2)
        self.assertListEqual([1, 90, 90,  4], list(x.shape))

    def test_down_sampling(self):
        x = np.ones((3, 100, 100, 18))
        x = down_sampling(x)
        self.assertListEqual([3, 46, 46, 36], list(x.shape))

    def test_up_sampling(self):
        x1 = np.ones((3, 150, 150, 256))
        x2 = np.ones((3, 70, 70, 512))
        conn = skip_connector(x1)
        x = up_sampling(x2, conn)
        self.assertListEqual([3, 136, 136, 256], list(x.shape))

    def test_u_net(self):
        model = u_net(input_shape=[284, 284, 3])
        inputs = tf.random.normal(shape=(3, 284, 284, 3))
        outputs = model(inputs)

        print('outputs.shape : ', outputs[1].shape)
