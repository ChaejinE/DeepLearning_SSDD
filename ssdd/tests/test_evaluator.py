from unittest import TestCase
from tf_2.segmentation.ssdd.eval.evaluator import Evaluator
import os
import numpy as np
import sys
import tensorflow as tf
import logging
import shutil
import cv2


class TestEvaluator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = {'segnet_h5_path': None,
                      'decnet_h5_path': None,
                      'test_dir': '',
                      'save_image_dir': '',
                      'label_key_name': '_label',
                      'resize_hw': [704, 256]}

        cls.evaluator = Evaluator(cls.config)

    def setUp(self) -> None:
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s][%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            stream=sys.stdout)

    def test_make_result_image_dir(self):
        # 지정된 Directory 에 Directory 들이 정상적으로 만들어 지는지 확인하는 test
        test_dir = '/tmp/test'
        self.evaluator._save_image_dir = test_dir
        self.evaluator._pos_dir = os.path.join(test_dir, 'pos')
        self.evaluator._neg_dir = os.path.join(test_dir, 'neg')
        self.evaluator._true_positive_save_dir = os.path.join(test_dir, 'pos', 'true_positive')
        self.evaluator._false_positive_save_dir = os.path.join(test_dir, 'pos', 'false_positive')
        self.evaluator._true_negative_save_dir = os.path.join(test_dir, 'neg', 'true_negative')
        self.evaluator._false_negative_save_dir = os.path.join(test_dir, 'neg', 'false_negative')

        self.evaluator.make_result_image_dir()

        self.assertTrue(os.path.isdir(self.evaluator._save_image_dir))
        self.assertTrue(os.path.isdir(self.evaluator._pos_dir))
        self.assertTrue(os.path.isdir(self.evaluator._neg_dir))
        self.assertTrue(os.path.isdir(self.evaluator._true_positive_save_dir))
        self.assertTrue(os.path.isdir(self.evaluator._false_positive_save_dir))
        self.assertTrue(os.path.isdir(self.evaluator._true_negative_save_dir))
        self.assertTrue(os.path.isdir(self.evaluator._false_negative_save_dir))

        shutil.rmtree(test_dir, ignore_errors=True)

    def test_make_gray_concat_image(self):
        # 예상동작 확인 test 1 - width 기준
        image = tf.random.uniform(shape=(1, 10, 10, 1))
        mask = tf.random.uniform(shape=(1, 10, 10, 3))
        pred = tf.random.uniform(shape=(1, 10, 10, 1))
        concat_image = self.evaluator.make_gray_concat_image(image.numpy()[0],
                                                             mask.numpy()[0],
                                                             pred.numpy()[0],
                                                             sort_axis=1)
        expected_shape = (10, 30, 1)
        self.assertEqual(expected_shape, np.shape(concat_image))

        # 예상동작 확인 test 2 - height 기준
        image = tf.random.uniform(shape=(1, 10, 10, 1))
        mask = tf.random.uniform(shape=(1, 10, 10, 3))
        pred = tf.random.uniform(shape=(1, 10, 10, 1))
        concat_image = self.evaluator.make_gray_concat_image(image.numpy()[0],
                                                             mask.numpy()[0],
                                                             pred.numpy()[0],
                                                             sort_axis=0)
        expected_shape = (30, 10, 1)
        self.assertEqual(expected_shape, np.shape(concat_image))

        # 전부 채널이 3일 때 test
        image = tf.random.uniform(shape=(1, 10, 10, 3))
        mask = tf.random.uniform(shape=(1, 10, 10, 3))
        pred = tf.random.uniform(shape=(1, 10, 10, 3))
        concat_image = self.evaluator.make_gray_concat_image(image.numpy()[0],
                                                             mask.numpy()[0],
                                                             pred.numpy()[0],
                                                             sort_axis=1)
        expected_shape = (10, 30, 1)
        self.assertEqual(expected_shape, np.shape(concat_image))

        # input type array 가 아닐 때 test 1
        image = 'np.random.random_sample(size=(10, 10, 3))'
        mask = tf.random.uniform(shape=(1, 10, 10, 3))
        pred = tf.random.uniform(shape=(1, 10, 10, 3))
        concat_image = self.evaluator.make_gray_concat_image(image,
                                                             mask.numpy()[0],
                                                             pred.numpy()[0],
                                                             sort_axis=1)
        self.assertEqual((10, 10, 1), np.shape(concat_image))

        # input type array 가 아닐 때 test 2
        image = None
        mask = tf.random.uniform(shape=(1, 10, 10, 3))
        pred = tf.random.uniform(shape=(1, 10, 10, 3))
        concat_image = self.evaluator.make_gray_concat_image(image,
                                                             mask.numpy()[0],
                                                             pred.numpy()[0],
                                                             sort_axis=1)
        self.assertEqual((10, 10, 1), np.shape(concat_image))

    def test_save_img_for_indicator(self):
        self.evaluator._save_image_dir = '/tmp/test'
        save_image = np.random.random_sample(size=(10, 10, 1))

        self.evaluator.make_result_image_dir()

        # true_positive 에 저장 되었는지 확인
        file_name = 'tp_image.jpg'
        self.evaluator.save_img_for_indicator('tp', file_name, save_image)
        check_tp_path = os.path.join(self.evaluator._true_positive_save_dir, file_name)
        self.assertTrue(os.path.exists(check_tp_path))

        # false_positive 에 저장 되었는지 확인
        file_name = 'fp_image.jpg'
        self.evaluator.save_img_for_indicator('fp', file_name, save_image)
        check_tp_path = os.path.join(self.evaluator._false_positive_save_dir, file_name)
        self.assertTrue(os.path.exists(check_tp_path))

        # true_negative 에 저장 되었는지 확인
        file_name = 'tn_image.jpg'
        self.evaluator.save_img_for_indicator('tn', file_name, save_image)
        check_tp_path = os.path.join(self.evaluator._true_negative_save_dir, file_name)
        self.assertTrue(os.path.exists(check_tp_path))

        # false_negative 에 저장 되었는지 확인
        file_name = 'fn_image.jpg'
        self.evaluator.save_img_for_indicator('fn', file_name, save_image)
        check_tp_path = os.path.join(self.evaluator._false_negative_save_dir, file_name)
        self.assertTrue(os.path.exists(check_tp_path))

        # input type 불량 시 logging.error 출력 되는지 확인 Test
        with self.assertRaises(Exception):
            self.evaluator.save_img_for_indicator(0, 1, save_image)
            self.evaluator.save_img_for_indicator(0, None, save_image)
            self.evaluator.save_img_for_indicator(0, 'string', save_image)
            self.evaluator.save_img_for_indicator(None, '', save_image)
            self.evaluator.save_img_for_indicator(0, 'a.jpg', save_image)
            self.evaluator.save_img_for_indicator(0, 'a.jpg', None)

        shutil.rmtree(self.evaluator._save_image_dir)

    def build_seg_model_for_test(self):
        inp = tf.keras.layers.Input(shape=(self.config['resize_hw'][0], self.config['resize_hw'][1], 1))
        some_feature = tf.keras.layers.Lambda(lambda x: tf.random.uniform(shape=(self.config['resize_hw'][0]//8,
                                                                                 self.config['resize_hw'][1]//8,
                                                                                 5)))(inp)
        some_seg_mask = tf.keras.layers.Lambda(lambda x: tf.random.uniform(shape=(self.config['resize_hw'][0]//8,
                                                                                  self.config['resize_hw'][1]//8,
                                                                                  1)))(inp)
        return tf.keras.Model(inputs=inp, outputs=[some_feature, some_seg_mask])

    def build_dec_model_for_test(self, pred):
        value = [0, 1] if pred == 'pos' else [1, 0]
        inp = tf.keras.layers.Input(shape=(self.config['resize_hw'][0]//8, self.config['resize_hw'][1]//8, 1))
        inp2 = tf.keras.layers.Input(shape=(self.config['resize_hw'][0]//8, self.config['resize_hw'][1]//8, 5))
        some_logit = tf.keras.layers.Lambda(lambda x: tf.constant(value))([inp, inp2])

        return tf.keras.Model(inputs=[inp, inp2], outputs=tf.cast(some_logit, dtype=tf.float32))

    def test_eval_decision_net(self):
        self.evaluator.test_dir = '/tmp/test'
        self.evaluator.seg_model = self.build_seg_model_for_test()

        os.makedirs(self.evaluator.test_dir, exist_ok=True)
        pos_dir, neg_dir = os.path.join(self.evaluator.test_dir, 'pos'), os.path.join(self.evaluator.test_dir, 'neg')
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
        for i in range(2):
            image = np.random.random_sample(size=(10, 10, 1))
            mask = np.random.random_sample(size=(10, 10, 1))
            cv2.imwrite(os.path.join(pos_dir, f'check_{i}_{i+2}.jpg'), image)
            cv2.imwrite(os.path.join(pos_dir, f'check_{i}_{i+2}_label.bmp'), mask)
            cv2.imwrite(os.path.join(neg_dir, f'check_{i}_{i+2}.jpg'), image)
            cv2.imwrite(os.path.join(neg_dir, f'check_{i}_{i+2}_label.bmp'), mask)

        # 전부 Positive 로 예측하는 상황 TEST
        self.evaluator.dec_model = self.build_dec_model_for_test(pred='pos')
        metric = self.evaluator.eval_decision_net()
        expected_count = [2, 0, 2, 0]
        count_list = [metric.tp, metric.tn, metric.fp, metric.fn]
        self.assertListEqual(expected_count, count_list)
        # 전부 Negative 로 예측하는 상황 TEST
        self.evaluator.dec_model = self.build_dec_model_for_test(pred='neg')
        metric.clear()

        metric = self.evaluator.eval_decision_net()
        expected_count = [0, 2, 0, 2]
        count_list = [metric.tp, metric.tn, metric.fp, metric.fn]
        self.assertListEqual(expected_count, count_list)
        metric.clear()

        # 정상 작동 test
        dir_name = '/tmp/test2'
        os.makedirs(dir_name, exist_ok=True)
        self.evaluator._save_image_dir = dir_name
        self.evaluator._pos_dir = os.path.join(self.evaluator._save_image_dir, 'pos')
        self.evaluator._neg_dir = os.path.join(self.evaluator._save_image_dir, 'neg')
        self.evaluator._true_positive_save_dir = os.path.join(self.evaluator._save_image_dir, 'pos', 'true_positive')
        self.evaluator._false_positive_save_dir = os.path.join(self.evaluator._save_image_dir, 'pos', 'false_positive')
        self.evaluator._true_negative_save_dir = os.path.join(self.evaluator._save_image_dir, 'neg', 'true_negative')
        self.evaluator._false_negative_save_dir = os.path.join(self.evaluator._save_image_dir, 'neg', 'false_negative')
        self.evaluator.dec_model = self.build_dec_model_for_test(pred='pos')
        metric = self.evaluator.eval_decision_net()

        expected_tp_count = len(os.listdir(os.path.join(dir_name, 'pos', 'true_positive')))
        expected_tn_count = len(os.listdir(os.path.join(dir_name, 'neg', 'true_negative')))
        expected_fp_count = len(os.listdir(os.path.join(dir_name, 'pos', 'false_positive')))
        expected_fn_count = len(os.listdir(os.path.join(dir_name, 'neg', 'false_negative')))
        self.assertEqual(expected_tp_count, metric.tp)
        self.assertEqual(expected_tn_count, metric.tn)
        self.assertEqual(expected_fp_count, metric.fp)
        self.assertEqual(expected_fn_count, metric.fn)
        expected_precision = metric.tp / (metric.tp + metric.fp)
        expected_recall = metric.tp / (metric.tp + metric.fn)
        expected_f1_score = 2 * (metric.precision * metric.recall) / (metric.precision + metric.recall)
        self.assertEqual(expected_precision, metric.precision)
        self.assertEqual(expected_recall, metric.recall)
        self.assertEqual(expected_f1_score, metric.f1_score)

        shutil.rmtree(dir_name, ignore_errors=True)
        shutil.rmtree(self.evaluator.test_dir, ignore_errors=True)

    def test_get_bbox_from_mask(self):
        def _assert_check(mask):
            bbox_dict = self.evaluator.get_bbox_from_mask(mask)

            self.assertEqual(dict, type(bbox_dict))
            for i in range(len(bbox_dict.items())):
                self.assertEqual(tuple, type(bbox_dict[i]))
                self.assertEqual(4, len(bbox_dict[i]))
                self.assertTrue(all(np.logical_and(np.array(bbox_dict[i][:2]) >= 0, np.array(bbox_dict[i][:2]) < 1)))
                self.assertTrue(all(np.logical_and(np.array(bbox_dict[i][2:]) > 0, np.array(bbox_dict[i][2:]) <= 1)))
                self.assertLess(bbox_dict[i][0], bbox_dict[i][2])
                self.assertLess(bbox_dict[i][1], bbox_dict[i][3])

        # TEST 1. 디펙 세 개 (정상 동작)
        mask = np.zeros(shape=(100, 100), dtype=np.uint8)
        mask[2:10, 3:7] = 255
        mask[21:42, 50:51] = 255
        mask[89:90, 89:90] = 255

        _assert_check(mask)

        # TEST 2. 아예 검정 이미지 (정상 동작)
        mask = np.zeros(shape=(100, 100), dtype=np.uint8)

        _assert_check(mask)

        # TEST 3. mask 가 float 이면? (raise !!)
        mask = np.ones(shape=(100, 100), dtype=np.uint8) / 255.
        self.assertRaises(Exception,
                          self.evaluator.get_bbox_from_mask,
                          mask=mask)

        # TEST 4. max 는 255 가 아니라 1 또는 이상한 숫자면? (0 or 나머지 숫자로 정상 동작함)
        mask = np.zeros(shape=(100, 100), dtype=np.uint8)
        mask[2:10, 3:7] = 1
        mask[21:42, 50:51] = 2
        mask[89:90, 89:90] = 3

        _assert_check(mask)

        # TEST 5. 이미지 전체가 255면? (정상 동작)
        mask = np.ones(shape=(100, 100), dtype=np.uint8) * 255

        _assert_check(mask)

    def test_classify_pred_bboxes(self):
        # TEST 1. 정상 동작
        label_bboxes = [(0.1, 0.1, 0.3, 0.3), (0.7, 0.7, 0.9, 0.9), (0.5, 0.5, 0.7, 0.7), (0.4, 0.4, 0.5, 0.5)]
        pred_bboxes = [(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7), (0.9, 0.9, 1.0, 1.0)]

        tp_bboxes, fp_bboxes, fn_bboxes = self.evaluator.classify_pred_bboxes(label_bboxes, pred_bboxes)

        self.assertListEqual([(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7)], tp_bboxes)
        self.assertListEqual([(0.9, 0.9, 1.0, 1.0)], fp_bboxes)
        self.assertListEqual([(0.7, 0.7, 0.9, 0.9), (0.4, 0.4, 0.5, 0.5)], fn_bboxes)

        # TEST 2. bboxes 타입이 리스트여도 되고 튜플이여도 됨
        label_bboxes = ((0.1, 0.1, 0.3, 0.3), (0.7, 0.7, 0.9, 0.9), (0.5, 0.5, 0.7, 0.7), (0.4, 0.4, 0.5, 0.5))
        pred_bboxes = ((0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7), (0.9, 0.9, 1.0, 1.0))

        tp_bboxes, fp_bboxes, fn_bboxes = self.evaluator.classify_pred_bboxes(label_bboxes, pred_bboxes)

        self.assertListEqual([(0.1, 0.1, 0.3, 0.3), (0.5, 0.5, 0.7, 0.7)], tp_bboxes)
        self.assertListEqual([(0.9, 0.9, 1.0, 1.0)], fp_bboxes)
        self.assertListEqual([(0.7, 0.7, 0.9, 0.9), (0.4, 0.4, 0.5, 0.5)], fn_bboxes)

        # TEST 3. bboxes 가 상대좌표이던 절대좌표이던 상관없음
        label_bboxes = [(0.1, 0.1, 0.3, 0.3), (7, 7, 9, 9)]
        pred_bboxes = [(0.1, 0.1, 0.3, 0.3), (7, 7, 9, 9)]

        tp_bboxes, fp_bboxes, fn_bboxes = self.evaluator.classify_pred_bboxes(label_bboxes, pred_bboxes)

        self.assertListEqual([(0.1, 0.1, 0.3, 0.3), (7., 7., 9., 9.)], tp_bboxes)
        self.assertListEqual([], fp_bboxes)
        self.assertListEqual([], fn_bboxes)

        # TEST 4. label bboxes 가 비어있으면?
        label_bboxes = []
        pred_bboxes = [(0.1, 0.1, 0.3, 0.3), (7, 7, 9, 9)]

        tp_bboxes, fp_bboxes, fn_bboxes = self.evaluator.classify_pred_bboxes(label_bboxes, pred_bboxes)
        self.assertListEqual([], tp_bboxes)
        self.assertListEqual(pred_bboxes, fp_bboxes)
        self.assertListEqual(label_bboxes, fn_bboxes)

        # TEST 5. pred bboxes 가 비어있으면?
        label_bboxes = [(0.1, 0.1, 0.3, 0.3), (7, 7, 9, 9)]
        pred_bboxes = []

        tp_bboxes, fp_bboxes, fn_bboxes = self.evaluator.classify_pred_bboxes(label_bboxes, pred_bboxes)

        self.assertListEqual([], tp_bboxes)
        self.assertListEqual(pred_bboxes, fp_bboxes)
        self.assertListEqual(label_bboxes, fn_bboxes)
