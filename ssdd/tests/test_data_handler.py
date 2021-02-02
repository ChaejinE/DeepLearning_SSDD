from tf_2.segmentation.ssdd.data.datasethandler import LensTubeDatasetHandler
import unittest
import os
import shutil
import numpy as np
import cv2
import logging
import sys


class TestDataHandler(unittest.TestCase):
    data_dir = '/tmp/test'

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(cls.data_dir, exist_ok=True)
        cls.train_dir = os.path.join(cls.data_dir, 'train')
        os.makedirs(cls.train_dir, exist_ok=True)
        train_positive_dir = os.path.join(cls.train_dir, 'pos')
        train_negative_dir = os.path.join(cls.train_dir, 'neg')
        os.makedirs(train_positive_dir, exist_ok=True)
        os.makedirs(train_negative_dir, exist_ok=True)
        train_image_dir = os.path.join(train_positive_dir, 'image')
        train_mask_dir = os.path.join(train_positive_dir, 'mask')
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_mask_dir, exist_ok=True)

        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

        for i in range(1, 4):
            image = np.random.random_sample(size=(10, 10, 1))
            mask = np.random.randint(low=0, high=2, size=(10, 10, 1))
            cv2.imwrite(os.path.join(train_positive_dir, 'check_{}.jpg'.format(i)), image)
            cv2.imwrite(os.path.join(train_positive_dir, 'check_{}_mask.bmp'.format(i)), mask)
            cv2.imwrite(os.path.join(train_negative_dir, 'check_{}.jpg'.format(i)), image)
            cv2.imwrite(os.path.join(train_negative_dir, 'check_{}_mask.bmp'.format(i)), mask)

            cv2.imwrite(os.path.join(train_image_dir, 'check_{}_{}.jpg'.format(i+2, i)), image)
            cv2.imwrite(os.path.join(train_mask_dir, 'check_{}_{}_mask.bmp'.format(i+2, i)), mask)
            cv2.imwrite(os.path.join(train_mask_dir, 'check_{}_{}_mask.bmp'.format(i+3, i+2)), mask)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.data_dir, ignore_errors=True)

    def test_get_matched_paths(self):
        h = 690
        w = 250
        label_key_name = '_mask'
        lenstube_handler = LensTubeDatasetHandler(self.train_dir,
                                                  resize_h_w=(h, w),
                                                  label_key_name=label_key_name)
        pos_paths = lenstube_handler.matched_positive_paths
        neg_paths = lenstube_handler.matched_negative_paths

        # class 내부에서 정상동작 했는지 확인 test
        # 1. mage 의 file 명이 mask file 명에 있는지 test
        [self.assertTrue(path_1.split('/')[-1].split('.')[0] in path_2.split('/')[-1].split('.')[0])
         for path_1, path_2 in pos_paths]
        [self.assertTrue(path_1.split('/')[-1].split('.')[0] in path_2.split('/')[-1].split('.')[0])
         for path_1, path_2 in neg_paths]

        # 2. image 와 mask 의 file 명이 정한 인덱스 까지 똑같은지 test
        [self.assertTrue(path_1.split('/')[-1].split('.')[0]+label_key_name == path_2.split('/')[-1].split('.')[0])
         for path_1, path_2 in pos_paths]
        [self.assertTrue(path_1.split('/')[-1].split('.')[0]+label_key_name == path_2.split('/')[-1].split('.')[0])
         for path_1, path_2 in neg_paths]

        # 다른 경로에 있어도 file 이름이 같은 것끼리 매칭해주는지 test
        tmp_image_dir = os.path.join(self.train_dir, 'pos', 'image')
        tmp_mask_dir = os.path.join(self.train_dir, 'pos', 'mask')
        tmp_paths = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_key='_mask')

        [self.assertTrue(path_1.split('/')[-1].split('.')[0]+label_key_name == path_2.split('/')[-1].split('.')[0])
         for path_1, path_2 in tmp_paths]

        # directory path type error Test
        path = lenstube_handler.get_matched_paths(None, tmp_mask_dir, matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(1., tmp_mask_dir, matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths('adsf', tmp_mask_dir, matching_key='_mask')
        self.assertEqual([], path)

        path = lenstube_handler.get_matched_paths(tmp_image_dir, None, matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir, 'adsf', matching_key='_mask')
        self.assertEqual([], path)
        # 빈 list 넣었을 때
        path = lenstube_handler.get_matched_paths([], tmp_mask_dir, matching_key='_mask')
        self.assertEqual([], path)

        # 확장명을 string 으로 안넣었 을 때 test
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, standard_ext=None, matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, standard_ext=1., matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_ext=None, matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_ext=1, matching_key='_mask')
        self.assertEqual([], path)

        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_key=None)
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_key=1)
        self.assertEqual([], path)

        # 매칭 되는게 없으면 빈 리스트 나오는지 test
        # key 가 일치하지 않을 때
        path = lenstube_handler.get_matched_paths(tmp_image_dir, tmp_mask_dir, matching_key='_label')
        self.assertEqual([], path)
        # 확장명을 원래와 다르게 입력했을 때
        path = lenstube_handler.get_matched_paths(tmp_image_dir,
                                                  tmp_mask_dir,
                                                  standard_ext='.bmp',
                                                  matching_key='_mask')
        self.assertEqual([], path)
        path = lenstube_handler.get_matched_paths(tmp_image_dir,
                                                  tmp_mask_dir,
                                                  matching_ext='.jpg',
                                                  matching_key='_mask')
        self.assertEqual([], path)

    def test_create_dataset(self):
        b = 1
        h = 690
        w = 250
        c = 1
        lenstube_handler = LensTubeDatasetHandler(self.train_dir, resize_h_w=(h, w), label_key_name='_mask')
        lenstube_handler.num_shuffle = True
        positive_dataset, negative_dataset = lenstube_handler.create_dataset()

        for img, mask, file_name in positive_dataset:
            self.assertEqual((b, h, w, c), np.shape(img))
            self.assertEqual((b, h//8, w//8, 1), np.shape(mask))

        for img, mask, file_name in negative_dataset:
            self.assertEqual((b, h, w, c), np.shape(img))
            self.assertEqual((b, h//8, w//8, 1), np.shape(mask))

        lenstube_handler.matched_positive_paths = list()
        with self.assertRaises(IndexError):
            positive_dataset, _ = lenstube_handler.create_dataset()
            for img, mask, file_name in positive_dataset:
                print(img)
                print(mask)
                print(file_name)
