import numpy as np


def is_relative_coords(coords):

    return not(np.array_equal(np.array(coords), np.array(coords, dtype=np.int64)))


def transform_abs_coord_to_relative(coords, origin_wh):
    """ 절대좌표를 상대좌표로 변환
    :param coords:      type list or numpy array
                        shape (4,) or (#coords, 4)

    :param origin_wh:   type list
                        shape (2,)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
    """
    coords = np.array(coords)

    coords[..., :2] = np.maximum(coords[..., :2], [0., 0.])
    coords[..., 2:] = np.minimum(coords[..., 2:], origin_wh)

    coords = coords[:, :] / (origin_wh * 2)

    return coords


def _transform_relative_coords_to_abs(coords, origin_wh):
    """상대좌표를 절대좌표로 변환

    :param coords:      type list or numpy array
                        shape (4,) or (#coords, 4)
                        range [0., 1.]

    :param origin_wh:   type list
                        shape (2,)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
    """
    coords = np.array(coords)
    coords[..., :2] = np.maximum(coords[..., :2], [0., 0.])
    coords[..., 2:] = np.minimum(coords[..., 2:], [1., 1.])

    coords = coords[..., :] * (origin_wh * 2)

    return coords


def transform_relative_ltrb_to_abs(coords, origin_wh):
    """상대좌표 ltrb 를 절대좌표로 변환

    :param coords:      type list or numpy array
                        shape (4,) or (#coords, 4)
                        range [0., 1.]
                        coord : (x_min, y_min, x_max, y_max)

    :param origin_wh:   type list
                        shape (2,)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
    """
    coords = np.array(coords)

    assert np.less(coords[..., :2], coords[..., 2:]).all(), 'left, top 값이 right, bottom 보다 클 수 없음.'

    coords = _transform_relative_coords_to_abs(coords, origin_wh)

    return coords


def transform_relative_ltwh_to_abs(coords, origin_wh):
    """상대좌표 ltwh 를 절대좌표로 변환

    :param coords:      type list or numpy array
                        shape (4,) or (#coords, 4)
                        range [0., 1.]
                        coord : (x_min, y_min, width, height)

    :param origin_wh:   type list
                        shape (2,)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
    """
    coords = _transform_relative_coords_to_abs(coords, origin_wh)

    coords[..., 2] = min(coords[..., 2], origin_wh[0]-coords[:, 0])
    coords[..., 3] = min(coords[..., 3], origin_wh[1]-coords[:, 1])

    return coords


def transform_corner_to_center(coords):
    """
    :param coords:  type list or numpy array
                    shape (4,) or (#coords, 4)
                    coord : (x_min, y_min, x_max, y_max)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
                coord : (center_x, center_y, width, height)
    """
    coords = np.array(coords)
    center_coords = np.concatenate([
        (coords[..., :2] + coords[..., 2:]) / 2,
        coords[..., 2:] - coords[..., :2]], axis=-1)

    return center_coords


def transform_center_to_corner(coords):
    """

    :param coords:  type list or numpy array
                    shape (4,) or (#coords, 4)
                    coord : (center_x, center_y, width, height)

    :return:    type numpy array
                shape (4,) or (#coords, 4)
                coord : (x_min, y_min, x_max, y_max)
    """
    coords = np.array(coords)
    corner_coords = np.concatenate([
        coords[..., :2] - coords[..., 2:] / 2,
        coords[..., :2] + coords[..., 2:] / 2], axis=-1)

    return corner_coords
