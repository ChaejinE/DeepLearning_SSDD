import tensorflow as tf
import os
import logging
import numpy as np
from typing import Tuple


class LensTubeDatasetHandler:
    def __init__(self, dataset_dir, resize_h_w,
                 num_shuffle=None, rot=False, flip=False, brightness=False,
                 image_format='.jpg', mask_format='.bmp', label_key_name='_label', mask_resize_h_w=None, batch_size=1):
        self.dataset_dir = dataset_dir
        self.resize_h_w = resize_h_w
        self.num_shuffle = num_shuffle
        self.image_format = image_format
        self.mask_format = mask_format
        self.rot = rot
        self.flip = flip
        self.brightness = brightness
        self.label_key_name = label_key_name
        self.matched_positive_paths, self.matched_negative_paths = self.init_matched_paths()
        self.mask_resize_h_w = mask_resize_h_w
        self.batch_size = batch_size

    @staticmethod
    def get_matched_paths(standard_dir, matching_dir, standard_ext='.jpg', matching_ext='.bmp', matching_key=''):
        """
        기준 디렉토리 의 file 들과 다른 directory 의 file 들을 matching 해주는 함수입니다.
        ex) input : ['/root/dir/image, ...], ['/root/dir/label, ...],
                    '.jpg', '.bmp', '_label'
        ex) output : [('/root/dir/image/file_name0.jpg', '/root/dir/label/file_name0_label.bmp'), ...]
        :param standard_dir: matching 기준 directory Type string dir path
        :param matching_dir: standard_dir 의 file 들과 matching 할 file 들의 상위 directroy Type string dir path
        :param standard_ext: standard_dir 의 files 확장자 명 Type string
        :param matching_ext: matching_dir 의 files 확장자 명 Type string
        :param matching_key: standard_dir file name 에 추가될 string (추가될 떄 matching_dir file 이름과 동일해야함.) Type string
        :return: matched_path Type list (elements is tuple)
        """
        try:
            standard_paths = [os.path.join(standard_dir, path) for path in os.listdir(standard_dir)
                              if path.endswith(standard_ext)]

            matched_paths = []
            for standard_path in standard_paths:
                expected_matching_path = os.path.join(matching_dir,
                                                      os.path.splitext(os.path.basename(standard_path))[0] +
                                                      matching_key +
                                                      matching_ext)

                if os.path.exists(expected_matching_path):
                    matched_paths.append((standard_path, expected_matching_path))

            if not matched_paths:
                logging.warning('Matching 되는 Mask Image 가  없습니다.')

            return matched_paths
        except Exception as e:
            logging.error('{}\n'
                          '[INPUT TYPE CHECK]\n'
                          'condition 1. standard_dir & matching_dir 은 string type 의 directory 경로\n'
                          '-> input standard_dir type : {}, matching_dir type : {}\n'
                          'condition 2. standard_ext & matching_ext 은 string type 의 확장명\n'
                          '-> input standard_ext type : {}, matching_ext type: {}\n'
                          'condition 3. matching_key 는 string type \n'
                          '-> input matching_key type : {}'.format(e,
                                                                   type(standard_dir),
                                                                   type(matching_dir),
                                                                   type(standard_ext),
                                                                   type(matching_ext),
                                                                   type(matching_key)))
            return []

    def load_image(self, path, mask=False, rot=False, flip_up_down=False, flip_left_right=False, brightness=False, mask_resize_h_w=None):
        img = tf.io.read_file(path)

        if mask_resize_h_w is None:
            mask_resize_h = self.resize_h_w[0]//8
            mask_resize_w = self.resize_h_w[1]//8
        else:
            mask_resize_h = mask_resize_h_w[0]
            mask_resize_w = mask_resize_h_w[1]

        if mask:
            img = tf.io.decode_bmp(img, channels=3)
            img = tf.image.rgb_to_grayscale(img)

            img = tf.image.resize(img, (mask_resize_h, mask_resize_w), method='nearest')
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.math.ceil(img)
        else:
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.rgb_to_grayscale(img)

            img = tf.image.resize(img, (self.resize_h_w[0], self.resize_h_w[1]), method='nearest')
            img = tf.image.convert_image_dtype(img, tf.float32)

        if rot:
            img = tf.image.rot90(img)

        if flip_up_down:
            img = tf.image.flip_up_down(img)

        if flip_left_right:
            img = tf.image.flip_left_right(img)

        if brightness:
            img = tf.image.random_brightness(img, max_delta=0.08, seed=None)

        return img

    def init_matched_paths(self):
        pos_dir = os.path.join(self.dataset_dir, 'pos')
        neg_dir = os.path.join(self.dataset_dir, 'neg')

        matched_pos_paths = self.get_matched_paths(pos_dir,
                                                   pos_dir,
                                                   self.image_format,
                                                   self.mask_format,
                                                   matching_key=self.label_key_name)
        matched_neg_paths = self.get_matched_paths(neg_dir,
                                                   neg_dir,
                                                   self.image_format,
                                                   self.mask_format,
                                                   matching_key=self.label_key_name)

        return matched_pos_paths, matched_neg_paths

    def map_fn(self, path_1, path_2):
        if self.flip:
            flip_up_down = (tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int64) == 1)
            flip_left_right = (tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int64) == 1)
        else:
            flip_up_down = False
            flip_left_right = False
        image = self.load_image(path_1, rot=self.rot, flip_up_down=flip_up_down, flip_left_right=flip_left_right, brightness=self.brightness)
        mask = self.load_image(path_2, mask=True, rot=self.rot, flip_up_down=flip_up_down, flip_left_right=flip_left_right, mask_resize_h_w=self.mask_resize_h_w)
        file_name = tf.strings.split(path_1, sep='/')[-1]

        return image, mask, file_name

    def create_dataset(self, small_data: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        데이터셋 생성함수
        :param small_data: 더 적은 데이터셋 기준으로 학습할지에 대한 Flag, True 시 데이터셋 중 더 적은 데이터 갯수에 맞춰 데이터셋을 생성하게한다.
        :return: positive_dataset, negative_dataset
        """
        num_positive = len(self.matched_positive_paths)
        num_negative = len(self.matched_negative_paths)
        logging.info('\n ========================='
                     '\n Positive Dataset : {} 개'
                     '\n Negative Dataset : {} 개'
                     '\n ========================='
                     .format(num_positive, num_negative))
        assert isinstance(small_data, bool), 'small_data argument 는 bool 이어야합니다.'
        take_num = min(num_positive, num_negative) if small_data else max(num_positive, num_negative)
        repeat_num = (max(num_positive, num_negative) // take_num) + 1
        logging.info(f'\n take_num : {take_num}')
        logging.info(f'\n repeat_num : {repeat_num}')

        try:
            positive_dataset = tf.data.Dataset.from_tensor_slices((np.array(self.matched_positive_paths).T[0],
                                                                   np.array(self.matched_positive_paths).T[1]))
            positive_dataset = positive_dataset.map(self.map_fn).repeat(repeat_num).take(take_num)
            negative_dataset = tf.data.Dataset.from_tensor_slices((np.array(self.matched_negative_paths).T[0],
                                                                   np.array(self.matched_negative_paths).T[1]))
            negative_dataset = negative_dataset.map(self.map_fn).repeat(repeat_num).take(take_num)

            if self.num_shuffle is not None:
                positive_dataset = positive_dataset.shuffle(self.num_shuffle).batch(self.batch_size, drop_remainder=True)
                negative_dataset = negative_dataset.shuffle(self.num_shuffle).batch(self.batch_size, drop_remainder=True)
            else:
                positive_dataset = positive_dataset.batch(self.batch_size, drop_remainder=True)
                negative_dataset = negative_dataset.batch(self.batch_size, drop_remainder=True)

            logging.info('\n ==========================================='
                         '\n positive dataset, negative dataset 생성완료 !'
                         '\n ===========================================')

            return positive_dataset, negative_dataset

        except IndexError:
            logging.info('\n ==========================================='
                         '\n positive dataset, negative dataset 생성실패 ! '
                         '\n ===========================================')

            raise IndexError('\n [Dataset Size CHECK]\n'
                             'dataset 이 비어있습니다.\n'
                             'matched_positive_paths : {} 개,\n'
                             'matched_negative_paths : {} 개'
                             .format(len(self.matched_positive_paths), len(self.matched_negative_paths)))


# 수정사항 3. Kolecktor Dataset 사용하기위한 코드 개발
POSITIVE_KolektorSDD = [['5'], ['6'], ['2'], ['3'], ['5'], ['7'], ['1'], ['2'], ['6'], ['3'],
                        ['4'], ['5'], ['3'], ['7'], ['3'], ['5'], ['5'], ['3'], ['5'], ['4'],
                        ['5'], ['6'], ['6'], ['1'], ['4'], ['5'], ['0'], ['3'], ['0'], ['0'],
                        ['1'], ['2'], ['6'], ['0'], ['5'], ['3'], ['0'], ['0', '1'], ['6', '7'], ['5'],
                        ['7'], ['3'], ['1'], ['6'], ['3'], ['7'], ['2'], ['5'], ['2'], ['4']]


def load_image(path, resize_h_w, mask=False):
    img = tf.io.read_file(path)

    if mask:
        img = tf.io.decode_bmp(img, channels=1)
        img = tf.image.resize(img, (resize_h_w[0]//8, resize_h_w[1]//8), method='nearest')
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.math.ceil(img)
    else:
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (resize_h_w[0], resize_h_w[1]), method='nearest')
        img = tf.image.convert_image_dtype(img, tf.float32)

    return img


def map_fn(path_1, path_2):
    image = load_image(path_1, resize_h_w=(1408, 512))
    mask = load_image(path_2, resize_h_w=(1408, 512), mask=True)

    return image, mask


def build_kolektorsdd_dataset(image_dir, is_positive):
    sub_dirs = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
    label_paths = []
    input_paths = []

    for sub_dir in sub_dirs:
        dir_path = os.path.join(image_dir, sub_dir)
        dir_number = int(sub_dir.replace("kos", ""))

        image_names = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg")])

        positive_indexs = []
        for index_str in POSITIVE_KolektorSDD[dir_number-1]:
            positive_indexs.append(int(index_str))

        for image_i, image_name in enumerate(image_names):
            if is_positive == (image_i in positive_indexs):
                name = os.path.basename(image_name)
                mask_name = "{}_label.bmp".format(name[:-4])

                image_path = os.path.join(os.path.join(image_dir, sub_dir), name)
                mask_path = os.path.join(os.path.join(image_dir, sub_dir), mask_name)

                input_paths.append(image_path)
                label_paths.append(mask_path)

    dataset = tf.data.Dataset.from_tensor_slices((input_paths, label_paths))
    dataset = dataset.map(map_fn)
    dataset = dataset.shuffle(52).batch(1)

    return dataset
