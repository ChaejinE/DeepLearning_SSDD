import unittest
import numpy as np
from tf_2.segmentation.ssdd.model.segmentation_network import SegmentationNet


class TestSegmentationNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.image_h_w_c_size = (1408, 512, 1)

    def test_create_model(self):
        segnet_model = SegmentationNet().build_model(input_shape=self.image_h_w_c_size)
        images = np.random.random_sample(size=(3, 1408, 512, 1))
        output1, output2 = segnet_model(images)
        self.assertEqual((3, 176, 64, 1024), np.shape(output1))
        self.assertEqual((3, 176, 64, 1), np.shape(output2))
