from tf_2.segmentation.ssdd.metric.f1_score import calculate_f1_score
import unittest
import numpy as np
import logging
import sys


class TestCalculateF1Score(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def test_calculate_f1_score(self):
        # 정상동작 test
        precision = np.random.random_sample()
        recall = np.random.random_sample()
        expected = precision * recall / (precision + recall) * 2
        result = calculate_f1_score(precision, recall)
        self.assertEqual(expected, result)

        # 문자열 타입으로 입력했을 때 error test
        precision = 'string1'
        recall = 'string2'
        expected = None
        result = calculate_f1_score(precision, recall)
        self.assertEqual(expected, result)

        # None 타입으로 입력했을 때 error test
        precision = 2
        recall = None
        expected = None
        result = calculate_f1_score(precision, recall)
        self.assertEqual(expected, result)

        # precision recall 이 둘다 0 일떄 error test
        precision = 0
        recall = 0
        expected = 0
        result = calculate_f1_score(precision, recall)
        self.assertEqual(expected, result)

        # precision recall 중 음수가 있을 때 error test
        precision = 0
        recall = -1
        expected = None
        result = calculate_f1_score(precision, recall)
        self.assertEqual(expected, result)
