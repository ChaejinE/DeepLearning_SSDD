from tf_2.segmentation.ssdd.metric.f1_score import calculate_precision
import logging
import sys
import unittest


class TestCalculatePrecision(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def test_calculate_precision(self):
        # 정상 동작 test
        true_positive = 2
        false_negative = 1
        precision = calculate_precision(true_positive, false_negative)
        expected = true_positive / (true_positive + false_negative)
        self.assertEqual(expected, precision)

        # 문자열 타입 input 에러 test
        true_positive = 'sad'
        false_negative = 1
        precision = calculate_precision(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, precision)

        # None 타입 input 에러 test
        true_positive = None
        false_negative = 1
        precision = calculate_precision(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, precision)

        # ZeroDivision  에러 test
        true_positive = 0
        false_negative = 0
        precision = calculate_precision(true_positive, false_negative)
        expected = 0
        self.assertEqual(expected, precision)

        # 부호가 0 이상이 아닐 때 test
        true_positive = 2
        false_negative = -1
        precision = calculate_precision(true_positive, false_negative)
        expected = None
        self.assertEqual(expected, precision)
