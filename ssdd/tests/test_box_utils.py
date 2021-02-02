from unittest import TestCase
from tf_2.segmentation.ssdd.utils.box_utils import _transform_relative_coords_to_abs
from tf_2.segmentation.ssdd.utils.box_utils import *


class TestBoxUtils(TestCase):
    def test_is_relative_coords(self):
        # TEST 1. 정상 동작
        coords = [[0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]]
        self.assertTrue(is_relative_coords(coords))

        # TEST 2. absolute coords
        coords = [[1., 1., 2., 2.], [100., 100., 200., 200.]]
        self.assertFalse(is_relative_coords(coords))

        # TEST 3. relative coords
        coords = [[0., 0., 1., 1.]]
        self.assertFalse(is_relative_coords(coords))

        # TEST 4. relative_coords
        coords = [[0., 0.1, 1., 1.]]
        self.assertTrue(is_relative_coords(coords))

        # TEST 5. relative_coords
        coords = [[0., 0., 1., 1.], [0.1, 0.1, 1., 1.1]]
        self.assertTrue(is_relative_coords(coords))

    def test_transform_relative_coords_to_abs(self):
        # TEST 1. 정상 동작
        origin_wh = [500, 300]

        coords = [(0.1, 0.1, 0.7, 0.7)]
        coords = _transform_relative_coords_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[50.,  30., 350., 210.]]), coords))

        coords = (0.1, 0.1, 0.7, 0.7)
        coords = _transform_relative_coords_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([50.,  30., 350., 210.]), coords))

        # TEST 2. 범위 초과 처리
        coords = [(-0.1, 0.1, 0.7, 1.1)]
        origin_wh = [500, 300]
        coords = _transform_relative_coords_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[0., 30., 350., 300.]]), coords))

    def test_transform_relative_ltwh_to_abs(self):
        # TEST 1. 정상 동작
        coords = [(0.1, 0.1, 0.7, 0.7)]
        origin_wh = [500, 300]
        coords = transform_relative_ltwh_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[50.,  30., 350., 210.]]), coords))

        # TEST 2. 만약 w, h 가 너무 커서 이미지 사이즈를 초과하면, max 한계로 자동 맞춤
        coords = [(0.5, 0.5, 1.0, 1.0)]
        origin_wh = [500, 300]
        coords = transform_relative_ltwh_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[250., 150., 250., 150.]]), coords))

    def test_transform_relative_ltrb_to_abs(self):
        # TEST 1. 정상 동작
        coords = [(0.1, 0.1, 0.7, 0.7)]
        origin_wh = [500, 300]
        coords = transform_relative_ltrb_to_abs(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[50.,  30., 350., 210.]]), coords))

        # TEST 2. right, bottom 값이 left, top 보다 작으면?
        coords = [(0.9, 0.5, 0.1, 0.6)]
        origin_wh = [500, 300]

        self.assertRaises(AssertionError,
                          transform_relative_ltrb_to_abs,
                          coords=coords,
                          origin_wh=origin_wh)

    def test_transform_abs_coord_to_relative(self):
        # TEST 1. 정상 동작
        coords = [[50., 30., 100., 150.]]
        origin_wh = [500, 300]
        coords = transform_abs_coord_to_relative(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[0.1, 0.1, 0.2, 0.5]]), coords))

        # TEST 2. w,h 범위 초과하면 1 로 맞춰버리기
        coords = [[50., 30., 100., 400.]]
        origin_wh = [500, 300]
        coords = transform_abs_coord_to_relative(coords, origin_wh)

        self.assertTrue(np.array_equal(np.array([[0.1, 0.1, 0.2, 1.]]), coords))

    def test_transform_center_to_corner(self):
        # TEST 1. 정상 동작
        coords = [[5., 5., 10., 10.], [10., 10., 3., 3.]]
        coords = transform_center_to_corner(coords)

        self.assertTrue(np.array_equal(np.array([[0., 0., 10., 10.],
                                                 [8.5, 8.5, 11.5, 11.5]]),
                                       coords))

    def test_transform_corner_to_center(self):
        # TEST 1. 정상 동작
        coords = [[1., 1., 9., 9.], [13., 13., 15., 15.]]
        coords = transform_corner_to_center(coords)

        self.assertTrue(np.array_equal(np.array([[5., 5., 8., 8.],
                                                 [14., 14., 2., 2.]]),
                                       coords))

