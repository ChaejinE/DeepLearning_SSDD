from unittest import TestCase
from tf_2.segmentation.ssdd.train.train_ops import dec_train_fn
from tf_2.segmentation.ssdd.model.segmentation_network import SegmentationNet
from tf_2.segmentation.ssdd.model.decision_network import DecisionNet
from tf_2.segmentation.ssdd.train.optimizer_builder import build_optimizer
import tensorflow as tf


class TestTrainOps(TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_decnet_train_fn(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            segnet = SegmentationNet().build_model(input_shape=[704, 256, 1])
            decnet = DecisionNet().network(mask_shape=[1, 88, 32, 1], feature_shape=[1, 88, 32, 1024], num_class=2)
            optimizer = build_optimizer(name='sgd')
        inputs = tf.random.normal(shape=(1, 704, 256, 1))
        labels = tf.constant([1])
        train_fn = dec_train_fn(strategy, segnet, decnet, optimizer)
        loss = train_fn(inputs, labels)

        self.assertTrue(float, loss)


