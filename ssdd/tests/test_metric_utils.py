from unittest import TestCase
from tf_2.segmentation.ssdd.metric.f1_score import count_f1, update_f1


class TestMetricUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_count_f1(self):
        # Test 1 : 정상 동작하는 경우
        label = [0, 0, 1, 1]
        pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        tp, fp, fn = count_f1(label, pred)

        self.assertEqual((1, 1, 1), (tp, fp, fn))

        # Test 2: label 의 shape 이 잘못 된 경우
        pred = [[1, 0], [0, 1], [1, 0], [0, 1]]

        self.assertRaises(Exception,
                          count_f1,
                          label=[[0, 0, 1, 1]],
                          pred=pred)

        self.assertRaises(Exception,
                          count_f1,
                          label=[[0], [0], [1], [1]],
                          pred=pred)

        # Test 3: pred 의 shape 이 잘못 된 경우
        label = [0, 0, 1, 1]

        self.assertRaises(Exception,
                          count_f1,
                          label=label,
                          pred=[0, 0, 1, 1])

        self.assertRaises(Exception,
                          count_f1,
                          label=label,
                          pred=[[0], [0], [1], [1]])

        self.assertRaises(Exception,
                          count_f1,
                          label=label,
                          pred=[[0, 0, 1, 1]])

        self.assertRaises(Exception,
                          count_f1,
                          label=label,
                          pred=[[0, 1, 1],  [0, 1, 1], [0, 1, 1],  [0, 1, 1]])

        # Test 4: label 에 0, 1 이외의 값이 들어간 경우
        pred = [[1, 0], [0, 1], [1, 0], [0, 1]]
        self.assertRaises(Exception,
                          count_f1,
                          label=[2, 0, 1, 0],
                          pred=pred)

        # Test 5: int 값이 아닌 float 값으로 들어간 경우
        label = [0., 0., 1., 1.]
        pred = [[1., 0.], [1., 0.], [0., 1.], [0., 1.]]
        tp, fp, fn = count_f1(label, pred)

        self.assertEqual((2, 0, 0), (tp, fp, fn))

    def test_update_f1(self):
        label = [0, 0, 1, 1]
        pred = [[0, 1], [1, 0], [0, 1], [1, 0]]

        # Test 1: 정상 동작
        tp, fp, fn = 0, 0, 0
        tp, fp, fn = update_f1(tp, fp, fn, label, pred)

        self.assertEqual((1, 1, 1), (tp, fp, fn))

        # Test 2: tp, fp, fn 이 float 인 경우
        tp, fp, fn = 0., 0., 0.
        tp, fp, fn = update_f1(tp, fp, fn, label, pred)

        self.assertEqual((1, 1, 1), (tp, fp, fn))
