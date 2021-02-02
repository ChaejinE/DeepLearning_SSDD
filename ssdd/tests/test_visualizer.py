from tf_2.segmentation.ssdd.utils.visualizer import Visualizer
from tf_2.segmentation.ssdd.model.segmentation_network import SegmentationNet
from tf_2.segmentation.ssdd.model.decision_network import DecisionNet
from unittest import TestCase

import tensorflow as tf
import numpy as np


class TestVisualizer(TestCase):
    @classmethod
    def setUpClass(cls):
        """

        h5 모델을 사용하는 경우에는 다음 코드를 사용
        cls.decnet = tf.keras.models.load_model(decnet_h5_path)

        """
        cls.segnet = SegmentationNet().network(image_shape=[704, 256, 1])
        cls.decnet = DecisionNet().network(mask_shape=[1, 88, 32, 1], feature_shape=[1, 88, 32, 1024], num_class=2)

        cls.visualizer = Visualizer(cls.segnet, cls.decnet)

    def test_find_target_layer(self):
        # TEST 1. conv2d layer 가 있는 모델에서 자동 탐색
        layer_name = self.visualizer.find_target_layer()
        self.assertEqual('activation_12', layer_name)

    def test_grad_cam(self):
        image = tf.random.normal((1, 704, 256, 1))

        heat_map, mask, pred = self.visualizer.grad_cam(image)

        self.assertEqual((704, 256), heat_map.shape)
        self.assertTrue(np.any(heat_map <= 255))
        self.assertTrue(np.any(heat_map >= 0))

        self.assertEqual((704, 256), mask.shape)
        self.assertTrue(np.any(mask <= 1))
        self.assertTrue(np.any(mask >= 0))

        self.assertEqual(np.int64, type(pred))

    def test_overlay_heatmap(self):
        image = np.ones((244, 244, 3), dtype=np.uint8)
        heatmap, output = self.visualizer.overlay_heatmap(image, image)

        self.assertEqual(np.shape(output), np.shape(heatmap))
        # 해당 합수는 cv2 라이브러리를 그대로 사용하기 때문에 input 데이터에 대한 정상 동작만 확인
