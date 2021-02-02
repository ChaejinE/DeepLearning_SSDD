from tf_2.segmentation.ssdd.utils.box_utils import *
import numpy as np


def crop_with_bboxes(image, bboxes, min_wh=None, margin=0):
    """
    :param image:   numpy array
                    shape (h, w, ch) or (h, w)
                    ch : 3, 1

    :param bboxes:  type list
                    shape (#bbox, 4)
                    bbox : (x_min, y_min, x_max, y_max), 상대좌표 or 절대좌표

    :param min_wh:  bbox 크기가 너무 작으면 적용할 최소 사이즈
                    type list, [width, height]
                    width, height 는 pixel 단위이며, type int
                    default None

    :param margin:  bbox 의 width, height 에 여유를 주기 위한 값, 주어진 pixel 값 만큼 상하좌우로 늘려줌.
                    type int

    :return:    cropped_images: type list
                                shape (#bbox, h, w, ch)
                crop_bboxes:    type list
                                shape (#bbox, 4)
                                bbox : (x_min, y_min, x_max, y_max), 절대좌표
    """
    if len(np.shape(bboxes)) == 1:

        return [], []
    cropped_images = []
    image_hw = np.array(image).shape[:2]
    image_wh = tuple(reversed(list(image_hw)))

    if is_relative_coords(bboxes):
        bboxes = transform_relative_ltrb_to_abs(bboxes, image_wh)
    bboxes = transform_corner_to_center(bboxes)
    bboxes[..., 2:] = np.add(bboxes[..., 2:], [margin * 2, margin * 2])

    if min_wh is not None:
        bboxes[..., 2:] = np.maximum(bboxes[..., 2:], min_wh)

    bboxes = transform_center_to_corner(bboxes)
    bboxes[..., :2] = np.maximum(bboxes[..., :2], [0., 0.])
    bboxes[..., 2:] = np.minimum(bboxes[..., 2:], image_wh)

    crop_bboxes = bboxes.astype(np.uint64)

    for bbox in crop_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cropped_images.append(image[y_min:y_max, x_min:x_max, ...])

    return cropped_images, list(crop_bboxes)

