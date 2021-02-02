from tf_2.segmentation.ssdd.model.segmentation_network import SegmentationNet
from tf_2.segmentation.ssdd.model.decision_network import DecisionNet
from tf_2.segmentation.ssdd.model.classifier import ClassifierTF
from tf_2.segmentation.ssdd.data.datasethandler import LensTubeDatasetHandler
from tf_2.segmentation.ssdd.utils.image_utils import *
from utils.tf2.box_utils import bbox_overlap
from tf_2.segmentation.ssdd.metric.f1_score import calculate_precision, calculate_recall, calculate_f1_score
from tf_2.segmentation.ssdd.metric.metric import Metric

import tensorflow as tf
import cv2
import numpy as np
import os
import logging
import time
import argparse
import sys


class Evaluator:
    def __init__(self, config):
        self.config = config

        self.seg_model = None
        self.dec_model = None
        self.classifier = None

        self.metric = Metric()
        self.segnet_h5_path = self.parse_config('segnet_h5_path')
        self.decision_h5_path = self.parse_config('decnet_h5_path')
        self.classifier_pb_dir = self.parse_config('classifier_pb_dir')

        self.resize_hw = self.parse_config('resize_hw')
        self.test_dir = self.parse_config('test_dir')
        self.label_key_name = self.parse_config('label_key_name')
        self._save_image_dir = self.parse_config('save_image_dir')
        self.mask_resize_h_w = self.parse_config('mask_resize_h_w')

        self._true_positive_save_dir = os.path.join(self._save_image_dir, 'tp')
        self._false_positive_save_dir = os.path.join(self._save_image_dir, 'fp')
        self._true_negative_save_dir = os.path.join(self._save_image_dir, 'tn')
        self._false_negative_save_dir = os.path.join(self._save_image_dir, 'fn')

        self._build_model()

    def parse_config(self, key):
        if key not in self.config:

            return None
        return self.config[key]

    @staticmethod
    def load_jpeg(path, channel=1, resize_hw=None):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=channel)
        if resize_hw is not None:
            img = tf.image.resize(img, resize_hw, method='nearest')
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img[tf.newaxis, :, :, :]

        return img

    def _build_model(self):
        if self.segnet_h5_path is not None:
            self.seg_model = SegmentationNet().build_model(pretrained=self.segnet_h5_path, input_shape=None)

        if self.decision_h5_path is not None:
            self.dec_model = DecisionNet().build_model(pretrained=self.decision_h5_path)

        if self.classifier_pb_dir is not None:
            self.classifier = ClassifierTF({'pb_dir': self.classifier_pb_dir})

    def pred(self, image_path):
        image = self.load_jpeg(image_path, resize_hw=self.resize_hw)
        if self.segnet_h5_path is not None:
            _, result_image = self.seg_model.predict(image)

        else:
            # TODO : decision network 추가하기
            result_image = None

        return result_image

    def make_result_image_dir(self):
        """
        pos(including tp directory, fp directory) directory 와
        neg(including tn directory, fn directory 를 만들어주는 함수입니다.
        :return: 없음
        """
        save_dir_list = [self._save_image_dir,
                         self._true_positive_save_dir, self._false_positive_save_dir,
                         self._true_negative_save_dir, self._false_negative_save_dir]
        logging.info('\n================================================================================='
                     '\n|True Positive | False Positive | True Negative | False Negative| Directory 생성 중'
                     '\n=================================================================================')
        [os.makedirs(dir_name, exist_ok=True) for dir_name in save_dir_list]
        logging.info('\n==================================================================================='
                     '\n|True Positive | False Positive | True Negative | False Negative| Directory 생성 완료'
                     '\n===================================================================================')

    def save_concat_image(self, image, label_mask, pred_mask, indicator, file_name):
        concat_image = self.make_gray_concat_image(image.numpy()[0]*255.,
                                                   label_mask.numpy()[0]*255.,
                                                   pred_mask[0]*255.,
                                                   sort_axis=1)
        self.save_img_for_indicator(indicator, str(file_name.numpy()[0].decode('utf-8')), concat_image)

    def eval_decision_net(self):
        _save_image_fn = None
        _metric = self.metric
        if self._save_image_dir:
            self.make_result_image_dir()
            _save_image_fn = self.save_concat_image

        def _eval(dataset, positive=True):
            for image, label_mask, file_name in dataset:
                st = time.time()
                feature_maps, pred_mask = self.seg_model(image)
                pred = self.dec_model([pred_mask, feature_maps])
                logging.info(f'seg-dec inference time : {time.time() - st}')
                indicator = _metric.classify_and_update(y_true=positive, y_pred=bool(np.argmax(pred)))
                logging.info('tp : {} | fp : {} | tn : {} | fn : {}'.format(self.metric.tp, self.metric.fp, self.metric.tn, self.metric.fn))
                logging.info('Precision : {} | Recall : {}'.format(self.metric.precision, self.metric.recall))
                logging.info('F1-Score : {}'.format(self.metric.f1_score))
                if callable(_save_image_fn):
                    _save_image_fn(image, label_mask, pred_mask, indicator, file_name)

        data_handler = LensTubeDatasetHandler(self.test_dir,
                                              self.resize_hw,
                                              num_shuffle=300,
                                              label_key_name=self.label_key_name,
                                              mask_resize_h_w=self.mask_resize_h_w)
        pos_dataset, neg_dataset = data_handler.create_dataset()
        for dataset, is_positive in zip([pos_dataset, neg_dataset], [True, False]):
            dataset = dataset.take(min(len(data_handler.matched_positive_paths), len(data_handler.matched_negative_paths)))
            _eval(dataset, is_positive)

        return self.metric

    def save_img_for_indicator(self, standard: str, file_name: str, image: np.ndarray):
        try:
            standard_dict = {'tp': self._true_positive_save_dir,
                             'tn': self._true_negative_save_dir,
                             'fp': self._false_positive_save_dir,
                             'fn': self._false_negative_save_dir}

            cv2.imwrite(os.path.join(standard_dict[standard], file_name), image)

        except Exception as e:
            print(f'failed to save_img_for_indicator as {e}')

            raise

    def save_segnet_result_images(self, image_path, save_dir, show=False):
        if not os.path.exists(save_dir):

            assert print('save dir is not exist')
        pred_image = self.pred(image_path)
        pred_image = tf.math.sigmoid(pred_image)[0]
        pred_image = np.array(pred_image, dtype=np.float32)
        pred_image = cv2.resize(pred_image, (self.config['resize_hw'][1], self.config['resize_hw'][0]))

        origin_image = cv2.imread(image_path)
        origin_image = cv2.resize(origin_image, (self.config['resize_hw'][1], self.config['resize_hw'][0]))
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        origin_image = origin_image / 255.

        label_image = cv2.imread(image_path[:-4]+'_label.bmp')
        label_image = cv2.resize(label_image, (self.config['resize_hw'][1], self.config['resize_hw'][0]))
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
        label_image = label_image / 255.

        concat_image = np.concatenate([origin_image, pred_image, label_image], axis=1)
        if show:
            cv2.imshow('concat', concat_image)
            cv2.waitKey(0)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), concat_image * 255.)

    @staticmethod
    def get_bbox_from_mask(mask):
        # TODO : 리팩토링 ; return 하는 bbox 를 dict 말고 list 로 바꾸기 ; lenstube 프로젝트와 통일
        """
        :param mask: shape (h, w), gray channel, binary image (0 or not)
        :return: dict, {0: (0., 0., 1., 1.), ...}
        """
        try:
            h_abs, w_abs = np.shape(mask)
            bbox_dict = {}

            contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 0, 0), 1)
                relative_ltrb = tuple(np.array((x, y, x+w, y+h)) / ((w_abs, h_abs) * 2))
                bbox_dict[i] = relative_ltrb

            return bbox_dict
        except Exception as e:
            print('Failed to get bbox from mask.')
            print(e)

            raise

    @staticmethod
    def classify_pred_bboxes(label_bboxes, pred_bboxes, iou_thr=0.0001):
        """
        :param label_bboxes: list or tuple, [(x_min, y_min, x_max, y_max), (0.1, 0.2, 0.7, 0.8), (1., 1., 10., 10.), ...]
        :param pred_bboxes: list, [(x_min, y_min, x_max, y_max), (0.1, 0.2, 0.7, 0.8), (...), ...]
        :param iou_thr: float, default=0.0001
        :return: tp_list, fp_list, fn_list / 각 list 는 [(x_min, y_min, x_max, y_max), (0.1, 0.2, 0.7, 0.8), (1., 1., 10., 10.), ...]
        """
        if len(label_bboxes) == 0 or len(pred_bboxes) == 0:

            return [], pred_bboxes, label_bboxes

        tp_bboxes, fp_bboxes, fn_bboxes = [], [], []
        iou_map = bbox_overlap([label_bboxes], [pred_bboxes])[0]
        best_pred_iou = np.amax(iou_map, axis=1)
        best_label_iou = np.amax(iou_map, axis=0)
        best_pred_idx = np.argmax(iou_map, axis=1)

        for i, iou in enumerate(best_pred_iou):
            if iou >= iou_thr:
                tp_bboxes.append(pred_bboxes[best_pred_idx[i]])
            else:
                fn_bboxes.append(label_bboxes[i])

        for i, iou in enumerate(best_label_iou):
            if iou < iou_thr:
                fp_bboxes.append(pred_bboxes[i])

        return tp_bboxes, fp_bboxes, fn_bboxes

    @staticmethod
    def draw_result_bboxes(image, tp_bboxes, fp_bboxes, fn_bboxes, show=False):
        def _draw_box(bboxes, color):
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(drawed, (x_min, y_min), (x_max, y_max), color, 1)

        drawed = np.copy(image) / 255.
        if len(np.shape(image)) == 2:
            drawed = cv2.cvtColor(drawed, cv2.COLOR_GRAY2BGR)

        for bboxes, color in zip([tp_bboxes, fp_bboxes, fn_bboxes], [(0, 255, 0), (255, 0, 0), (0, 0, 255)]):
            _draw_box(bboxes, color)

        if show:
            cv2.imshow('drawed', drawed)
            cv2.waitKey(0)

        return drawed * 255.

    @staticmethod
    def make_gray_concat_image(origin_image, mask, pred_mask, sort_axis):
        """
        image, mask, pred_mask 를 정제하여 정렬 기준으로 concat image 를 만들어주는 함수입니다.
        :param origin_image: original image (h, w, c)
        :param mask: original label mask (h ,w ,c)
        :param pred_mask: predict mask (h, w, c)
        :param sort_axis: 정렬할 축 (0 : height, 1: width)
        :return: concat_image
        """
        try:
            mask = cv2.resize(np.float32(mask), (origin_image.shape[1], origin_image.shape[0]))
            mask = mask[:, :, np.newaxis] if len(mask.shape) == 2 else mask
            pred_mask = cv2.resize(np.float32(pred_mask), (origin_image.shape[1], origin_image.shape[0]))
            pred_mask = pred_mask[:, :, np.newaxis] if len(pred_mask.shape) == 2 else pred_mask

            image_list = [origin_image, mask, pred_mask]
            for idx, img in enumerate(image_list):
                if img.shape[-1] == 3:
                    image_list[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

            origin_image, mask, pred_mask = image_list
            concat_image = np.concatenate((origin_image, mask, pred_mask), axis=sort_axis)

            return concat_image

        except AttributeError:
            logging.error('[AttributeError] [INPUT TYPE CHECK]\n'
                          'images type 은 실수형 numpy array 입니다. -> origin_image : {}, mask : {}, pred_mask : {}'.
                          format(type(origin_image), type(mask), type(pred_mask)))

            return np.zeros(shape=(10, 10, 1))

    def create_result_image(self, image, label_mask, pred_mask, file_name=None, show=False, save=False):
        hw = np.shape(label_mask)[:2]
        image = cv2.resize(image, (hw[1], hw[0]))

        label_mask = cv2.cvtColor(label_mask, cv2.COLOR_GRAY2BGR)
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        concat_image = np.concatenate([image, label_mask, pred_mask], axis=1)
        if show:
            cv2.imshow('concat', concat_image / 255.)
            cv2.waitKey(0)
        if save and file_name is not None:
            cv2.imwrite(os.path.join(self._save_image_dir, file_name), concat_image)

    def eval_segmentation_net(self, iou_thr=0.0001, sigmoid_thr=0.5, save_image=False, show_image=False, save_crop=False, label='pos'):
        data_handler = LensTubeDatasetHandler(self.test_dir,
                                              self.resize_hw,
                                              num_shuffle=300,
                                              label_key_name=self.label_key_name,
                                              mask_resize_h_w=self.mask_resize_h_w)

        pos_dataset, neg_dataset = data_handler.create_dataset()

        dataset = pos_dataset
        nrof_dataset = data_handler.matched_positive_paths

        if label == 'neg':
            dataset = neg_dataset
            nrof_dataset = data_handler.matched_negative_paths

        total_label_num = 0
        tp = 0
        fp = 0
        fn = 0
        i = 1
        start_time = time.time()
        for image, label_mask, file_name in dataset:
            sys.stdout.write("\r{} / {}".format(i, len(nrof_dataset)))
            sys.stdout.flush()
            i += 1

            image = tf.cast(image, tf.float32)
            _, pred_mask = self.seg_model(image)

            image = image.numpy()[0] * 255.
            h, w = image.shape[:2]

            pred_mask = np.array(pred_mask[0], dtype=np.float32)
            pred_mask = cv2.resize(pred_mask, (w, h))
            pred_mask = tf.math.sigmoid(pred_mask).numpy()
            pred_mask = tf.where(tf.greater(pred_mask, sigmoid_thr), 1., 0.).numpy()
            pred_mask *= 255.

            label_mask = np.array(label_mask[0] * 255., dtype=np.float32)
            label_mask = cv2.resize(label_mask, (w, h))
            label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)

            label_bboxes = self.get_bbox_from_mask(np.array(label_mask, dtype=np.uint8))
            pred_bboxes = self.get_bbox_from_mask(np.array(pred_mask, dtype=np.uint8))

            origin_image_dir = os.path.join(self.test_dir, label)
            file_name = file_name.numpy()[0].decode('utf-8')
            origin_image = cv2.imread(os.path.join(origin_image_dir, file_name))

            if self.classifier is not None:
                pred_bboxes = self.classify_real_defects(origin_image, list(pred_bboxes.values()))
            else:
                pred_bboxes = pred_bboxes.values()

            total_label_num += len(label_bboxes)
            tp_bboxes, fp_bboxes, fn_bboxes = self.classify_pred_bboxes(list(label_bboxes.values()),
                                                                        list(pred_bboxes))
            tp += len(tp_bboxes)
            fp += len(fp_bboxes)
            fn += len(fn_bboxes)
            total_matched_num = tp

            print('\n\n======= 평가 결과 =======')
            print('sig_thr :', sigmoid_thr)
            print('total label num : ', total_label_num)
            print('total matched num : ', total_matched_num)
            print('total matched num / total label num : ', total_matched_num/total_label_num)
            print('tp :', tp)
            print('fp :', fp)
            print('fn :', fn)
            print('precision : ', calculate_precision(tp, fp))
            print('recall : ', calculate_recall(tp, fn))
            print('iou thr : ', iou_thr)
            print('time spent :', time.time() - start_time)

            tp_images, tp_bboxes = crop_with_bboxes(origin_image, tp_bboxes, min_wh=[20, 20], margin=5)
            fp_images, fp_bboxes = crop_with_bboxes(origin_image, fp_bboxes, min_wh=[20, 20], margin=5)
            fn_images, fn_bboxes = crop_with_bboxes(origin_image, fn_bboxes, min_wh=[20, 20], margin=5)
            drawed_image = self.draw_result_bboxes(origin_image, tp_bboxes, fp_bboxes, fn_bboxes)

            self.create_result_image(drawed_image, label_mask, pred_mask, file_name, show=show_image, save=save_image)

            if save_crop:
                for images, name in zip([tp_images, fp_images, fn_images], ['tp', 'fp', 'fn']):
                    self.save_bbox_images(images, file_name, os.path.join(self._save_image_dir, name), key_name=name)

    def classify_real_defects(self, image, boxes):
        if len(boxes) == 0:

            return []
        h, w = np.shape(image)[:2]
        if is_relative_coords(boxes):
            boxes = transform_relative_ltrb_to_abs(boxes, [w, h])
        boxes = self.classifier.classify(image, boxes)
        if len(boxes) == 0:

            return []
        return transform_abs_coord_to_relative(boxes, [w, h])

    @staticmethod
    def save_bbox_images(images, file_name, save_folder, key_name=''):
        for i, image in enumerate(images):
            sys.stdout.write("\r {} / {}".format(i, len(images)))
            sys.stdout.flush()

            save_path = os.path.join(save_folder, os.path.splitext(file_name)[0]+'_'+key_name+'_{}'.format(i)+'.jpg')
            cv2.imwrite(save_path, image)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="select : 1. 'seg', 2. 'dec'")
    parser.add_argument("--segnet-h5-path", default=None, help="/path/to/segnet_h5_file_path")
    parser.add_argument("--decnet-h5-path", default=None, help="/path/to/decnet_h5_file_path")
    parser.add_argument("--classifier-pb-dir", default=None, help="/path/to/classifier_pb_dir")
    parser.add_argument("--test-dir", required=True, help="/path/to/test_image_folder")
    parser.add_argument("--save-image-dir", required=True, help="/path/to/result_image_save_folder")
    parser.add_argument("--label-key-name", default="_label")
    parser.add_argument("--resize-hw", type=int, nargs='*', default=[704, 256], help="ex) --resize-hw 256 256")
    parser.add_argument("--save-image", default=False, help="save segmentation eval result image or not")
    parser.add_argument("--show-image", default=False)
    parser.add_argument("--save-crop", default=False)
    parser.add_argument("--iou-thr", type=float, default=0.0001)
    parser.add_argument("--sigmoid-thr", type=float, default=0.5)
    parser.add_argument("--mask-resize-h-w", type=int, default=None, nargs='*', help="--mask-resize-h-w 196 196")
    args = parser.parse_args()

    print(args.__dict__)
    evaluator = Evaluator(args.__dict__)

    if args.eval == 'seg':
        if args.segnet_h5_path is None:
            print('--segnet-h5-path 를 입력해주세요.')

            raise ValueError
        print('args.show_image : ', args.show_image)
        evaluator.eval_segmentation_net(iou_thr=args.iou_thr,
                                        sigmoid_thr=args.sigmoid_thr,
                                        save_image=args.save_image,
                                        show_image=args.show_image,
                                        save_crop=args.save_crop)
    elif args.eval == 'dec':
        if args.decnet_h5_path is None:
            print('--decnet-h5-path 를 입력해주세요.')

            raise ValueError
        evaluator.eval_decision_net()
    else:
        print("--eval 은 'seg' 또는 'dec' 중에서 선택해주세요.")

        raise ValueError

    """segnet eval 실행 코드 샘플
    python3 ./tf_2/object_detection/defect_detection/model/evaluator.py\
    --eval seg\
    --segnet-h5-path '/Users/hyeseong/deep_learning/saved_models/segnet_all_v4_gpu1_epoch_135.h5'\
    --classifier-pb-dir '/Users/hyeseong/deep_learning/saved_models/export_0907'\
    --test-dir '/Users/hyeseong/datasets/private/lens_tube/lens_tube_0728/lens_tube_0728_v5'\
    --save-image-dir '/Users/hyeseong/deep_learning/result_images/segnet_eval'
    """

    """decnet eval 실행 코드 샘플
    python3 ./tf_2/segmentation/ssdd/eval/evaluator.py\
    --eval dec\
    --segnet-h5-path '/Users/hyeseong/deep_learning/saved_models/segnet_lenstube_0728_v4_gpu3_epoch_36.h5'\
    --decnet-h5-path '/Users/hyeseong/deep_learning/saved_models/1598931728357_decnet_lenstube_0728_v4_epoch_3.h5'\
    --resize-hw 1408 512\
    --test-dir '/Users/hyeseong/datasets/private/lens_tube/lens_tube_0728/lens_tube_0728_v6'\
    --save-image-dir '/Users/hyeseong/deep_learning/result_images/decnet_eval'
    """
