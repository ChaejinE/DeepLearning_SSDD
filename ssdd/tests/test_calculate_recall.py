from tf_2.segmentation.ssdd.metric.f1_score import calculate_recall
import logging
import sys
import unittest


class TestCalculateRecall(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def test_calculate_recall(self):
        # 정상 동작 test
        true_positive = 2
        false_negative = 1
        recall = calculate_recall(true_positive, false_negative)
        expected = true_positive / (true_positive + false_negative)
        self.assertEqual(expected, recall)

        # 문자열 타입 input 에러 test
        true_positive = 'sad'
        false_negative = 1
        recall = calculate_recall(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, recall)

        # None 타입 input 에러 test
        true_positive = None
        false_negative = 1
        recall = calculate_recall(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, recall)

        # ZeroDivision  에러 test
        true_positive = 0
        false_negative = 0
        recall = calculate_recall(true_positive, false_negative)
        expected = 0
        self.assertEqual(expected, recall)

        # 부호가 0 이상이 아닐 때 test
        true_positive = 2
        false_negative = -1
        recall = calculate_recall(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, recall)
