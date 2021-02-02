from unittest import TestCase
from tf_2.segmentation.ssdd.utils.image_utils import *

import numpy as np


class TestImageUtils(TestCase):
    def test_crop_with_bboxes(self):
        # TEST 1. 정상 동작
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        bboxes = [(0.5, 0.5, 0.7, 0.7), (0.2, 0.2, 0.6, 0.6)]
        min_wh = None
        margin = 0
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[100, 50, 140, 70],
                                        [40, 20, 120, 60]],
                                       bboxes))
        for image, bbox in zip(images, bboxes):
            image_hw = list(image.shape[:2])
            bbox_hw = [bbox[3]-bbox[1], bbox[2]-bbox[0]]
            self.assertListEqual(image_hw, bbox_hw)

        # TEST 2. margin 적용
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        bboxes = [(0.5, 0.5, 0.7, 0.7), (0.2, 0.2, 0.6, 0.6)]
        min_wh = None
        margin = 6
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[94, 44, 146, 76],
                                        [34, 14, 126, 66]],
                                       bboxes))

        # TEST 3. min_wh 적용
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        bboxes = [(0.5, 0.5, 0.51, 0.51)]
        min_wh = [16, 16]
        margin = 0
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[93, 42, 109, 58]],
                                       bboxes))

        # TEST 4. min_wh 가 이미지 크기보다 크게 들어가면? (이미지 사이즈로 맞춰서 짜름)
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        bboxes = [(0.5, 0.5, 0.51, 0.51)]
        min_wh = [201, 101]
        margin = 0
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[0, 0, 200, 100]],
                                       bboxes))

        # TEST 5. 이미지 채널 다양하게 테스트
        bboxes = [(0.5, 0.5, 0.7, 0.7), (0.2, 0.2, 0.6, 0.6)]
        min_wh = None
        margin = 0

        image = np.zeros((100, 200, 1), dtype=np.uint8)
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[100, 50, 140, 70],
                                        [40, 20, 120, 60]],
                                       bboxes))
        for image, bbox in zip(images, bboxes):
            image_hw = list(image.shape[:2])
            bbox_hw = [bbox[3]-bbox[1], bbox[2]-bbox[0]]
            self.assertListEqual(image_hw, bbox_hw)

        image = np.zeros((100, 200), dtype=np.uint8)
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([[100, 50, 140, 70],
                                        [40, 20, 120, 60]],
                                       bboxes))
        for image, bbox in zip(images, bboxes):
            image_hw = list(image.shape[:2])
            bbox_hw = [bbox[3]-bbox[1], bbox[2]-bbox[0]]
            self.assertListEqual(image_hw, bbox_hw)

        # TEST 6. bboxes 가 비어있으면?
        bboxes = []
        min_wh = None
        margin = 0

        image = np.zeros((100, 200, 1), dtype=np.uint8)
        images, bboxes = crop_with_bboxes(image, bboxes, min_wh, margin)

        self.assertTrue(np.array_equal([], bboxes))
