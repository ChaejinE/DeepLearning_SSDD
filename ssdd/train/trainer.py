from tf_2.segmentation.ssdd.train.train_ops import seg_train_fn, seg_valid_fn, dec_train_fn, dec_valid_fn, segdec_train_fn, segdec_valid_fn
from tf_2.segmentation.ssdd.model.segmentation_network import SegmentationNet
from tf_2.segmentation.ssdd.model.decision_network import DecisionNet
from tf_2.segmentation.ssdd.model.u_net import u_net
from tf_2.segmentation.ssdd.data.datasethandler import LensTubeDatasetHandler
from tf_2.segmentation.ssdd.train.optimizer_builder import build_optimizer
from tf_2.segmentation.ssdd.utils.early_stop import EarlyStopping
from pytz import timezone
from datetime import datetime

import tensorflow as tf
import numpy as np
import os


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
        self.num_replicas = self.strategy.num_replicas_in_sync
        print(f'GPU 장치 수 : {self.num_replicas}')

        self.learning_name = self.parse_config('learning_name')

        self.train_dataset_dir = os.path.join(self.parse_config('dataset_path'), 'train')
        self.valid_dataset_dir = os.path.join(self.parse_config('dataset_path'), 'valid')

        self.image_h_w_c = self.parse_config('image_h_w_c')
        self.num_shuffle = self.parse_config('num_shuffle')
        self.rot = self.parse_config('rot')
        self.flip = self.parse_config('flip')
        self.brightness = self.parse_config('brightness')
        self.mask_resize_h_w = self.parse_config('mask_resize_h_w')
        self.small_data: bool = False if self.parse_config('small_data') is None else self.parse_config('small_data')

        self.epochs = self.parse_config('epochs')
        self.batch_size = self.parse_config('batch_size')

        self.save_dir = self.parse_config('saved_model_path')
        self.save_model_per_epoch = self.parse_config('save_model_per_epoch')

        os.makedirs(os.path.join(self.save_dir, 'metafiles'), exist_ok=True)
        self.metafile_path = os.path.join(self.save_dir, 'metafiles/{}_{}_epoch_{}.h5')

        self.train_writer_path = os.path.join(self.save_dir, f'eventfiles/{self.learning_name}_train_{datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")}')
        self.valid_writer_path = os.path.join(self.save_dir, f'eventfiles/{self.learning_name}_valid_{datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%dT%H:%M:%S")}')

        self.train_summary_writer = tf.summary.create_file_writer(self.train_writer_path)
        self.valid_summary_writer = tf.summary.create_file_writer(self.valid_writer_path)

        self.num_tensorboard_images = self.parse_config('num_tensorboard_images')

        self.early_stop: bool = self.parse_config('early_stop')
        self.early_stop_patience: int = self.parse_config('early_stop_patience')
        self.early_stop_target_grad: float = self.parse_config('early_stop_target_grad')

        self.train_pos_stopper, self.train_neg_stopper, self.valid_pos_stopper, self.valid_neg_stopper \
            = [EarlyStopping(patience=self.early_stop_patience,
                             decrease_target_grad=self.early_stop_target_grad) for _ in range(4)]

        self.dataset = None
        self.train_fn = None
        self.valid_fn = None
        self.optimizer = None
        self.segnet = None
        self.decnet = None

    def parse_config(self, key):
        if key not in self.cfg:

            return None
        return self.cfg[key]

    def create_distributed_dataset(self):
        train_dataset_handler = LensTubeDatasetHandler(self.train_dataset_dir,
                                                       self.image_h_w_c,
                                                       self.num_shuffle,
                                                       rot=self.rot,
                                                       flip=self.flip,
                                                       brightness=self.brightness,
                                                       mask_resize_h_w=self.mask_resize_h_w,
                                                       batch_size=self.batch_size)

        valid_dataset_handler = LensTubeDatasetHandler(self.valid_dataset_dir,
                                                       self.image_h_w_c,
                                                       mask_resize_h_w=self.mask_resize_h_w,
                                                       batch_size=self.batch_size)

        train_pos_data, train_neg_data = train_dataset_handler.create_dataset(self.small_data)
        valid_pos_data, valid_neg_data = valid_dataset_handler.create_dataset(self.small_data)

        train_pos_data, train_neg_data, valid_pos_data, valid_neg_data = \
            tuple(map(lambda x: self.strategy.experimental_distribute_dataset(x),
                      [train_pos_data, train_neg_data, valid_pos_data, valid_neg_data]))

        return [(train_pos_data, train_neg_data), (valid_pos_data, valid_neg_data)]

    @classmethod
    def get_result_image(cls, image, label_mask, pred_mask):
        h, w = image.shape[1:3]

        image, label_mask, pred_mask = tuple(map(lambda x: tf.cast(x, tf.float32), [image, label_mask, pred_mask]))
        pred_mask, label_mask = tuple(map(lambda x: tf.image.resize(x, (h, w)), [pred_mask, label_mask]))
        pred_mask = tf.math.sigmoid(pred_mask)
        concat_image = tf.concat([image, tf.image.grayscale_to_rgb(label_mask), tf.image.grayscale_to_rgb(pred_mask)], axis=2)

        return concat_image

    def save_model(self, epoch, **model):
        if epoch % self.save_model_per_epoch == 0:
            now_time = datetime.now(timezone('Asia/Seoul')).strftime("%Y-%m-%dT%H:%M:%S")
            for _k, _v in model.items():
                save_path = self.metafile_path.format(now_time, self.learning_name+'_'+_k, epoch)
                _v.save(save_path)

    def tensorboard_images(self, epoch, summary_writer, i):
        dataset_dir = [self.train_dataset_dir, self.valid_dataset_dir][i]
        dataset_handler = LensTubeDatasetHandler(dataset_dir,
                                                 self.image_h_w_c,
                                                 mask_resize_h_w=self.mask_resize_h_w,
                                                 batch_size=1)

        pos_datasets, neg_datasets = dataset_handler.create_dataset(small_data=True)
        pos_datasets, neg_datasets = tuple(map(lambda x: x.take(self.num_tensorboard_images), [pos_datasets, neg_datasets]))
        for key, datasets in zip(['pos', 'neg'], [pos_datasets, neg_datasets]):
            for step, (image, mask, _) in enumerate(datasets):
                _, pred_mask = self.segnet(image)
                result_img = self.get_result_image(image, mask, pred_mask)
                with summary_writer.as_default():
                    tf.summary.image(f'{key}_image_{step}', result_img, step=epoch)

    def tensorboard_scalars(self, epoch, summary_writer, **kwargs):
        with summary_writer.as_default():
            for _k, _v in kwargs.items():
                tf.summary.scalar(_k, _v, step=epoch)

    def label_fn(self, label):

        return label

    def save_fn(self, epoch):
        pass

    def early_stop_fn(self, pos_loss, neg_loss, i):
        key = ["train", "valid"][i]
        pos_stopper = [self.train_pos_stopper, self.valid_pos_stopper][i]
        neg_stopper = [self.train_neg_stopper, self.valid_neg_stopper][i]
        if self.early_stop:
            if pos_stopper.check_state(pos_loss) or neg_stopper.check_state(neg_loss):
                print(f'{key} Early Stopping : target loss delta <{self.early_stop_target_grad}>'
                      f' 에 도달하여 학습을 종료합니다.')

                raise Exception()

    def train(self):
        for epoch in range(self.epochs):
            for i, (pos_data, neg_data) in enumerate(self.dataset):
                avg_pos_loss = 0
                avg_neg_loss = 0

                loss_fn = [self.train_fn, self.valid_fn][i]
                summary_writer = [self.train_summary_writer, self.valid_summary_writer][i]
                for step, ((pos_image, pos_mask, _), (neg_image, neg_mask, _)) in enumerate(zip(pos_data, neg_data)):
                    pos_loss = loss_fn(pos_image, self.label_fn(pos_mask))
                    neg_loss = loss_fn(neg_image, self.label_fn(neg_mask))

                    avg_pos_loss = (avg_pos_loss * step + pos_loss) / (step+1)
                    avg_neg_loss = (avg_neg_loss * step + neg_loss) / (step+1)

                    print(f'| {["train", "valid"][i]} | epoch : {epoch} | step : {step} '
                          f'| avg_pos_loss : {avg_pos_loss} | avg_neg_loss : {avg_neg_loss} |')

                self.tensorboard_scalars(epoch, summary_writer,
                                         pos_loss=avg_pos_loss, neg_loss=avg_neg_loss)
                self.tensorboard_images(epoch, summary_writer, i)
                self.early_stop_fn(avg_pos_loss, avg_neg_loss, i)

            if callable(self.save_fn):
                self.save_fn(epoch)


class SegTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.weight = self.parse_config('weight')
        self.pretrained_path = os.path.join(self.save_dir, self.parse_config("pretrained_segnet")) if self.parse_config("pretrained_segnet") is not None else None

        self.get_ready()

    def get_ready(self):
        self.dataset = self.create_distributed_dataset()
        with self.strategy.scope():
            self.segnet = SegmentationNet().build_model(pretrained=self.pretrained_path,
                                                       input_shape=self.image_h_w_c)
            self.optimizer = build_optimizer(self.parse_config('optimizer'), lr=self.parse_config('lr'))
        self.train_fn = seg_train_fn(self.strategy, self.segnet, self.optimizer, self.weight)
        self.valid_fn = seg_valid_fn(self.strategy, self.segnet, self.weight)

    def save_fn(self, epoch):
        self.save_model(epoch, seg=self.segnet)


class DecTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.mask_shape = self.parse_config('mask_shape')
        self.feature_shape = self.parse_config('feature_shape')
        self.num_class = self.parse_config('num_class')
        self.pretrained_segnet_path = os.path.join(self.save_dir, self.parse_config("pretrained_segnet")) if self.parse_config("pretrained_segnet") is not None else None
        self.pretrained_decnet_path = os.path.join(self.save_dir, self.parse_config("pretrained_decnet")) if self.parse_config("pretrained_decnet") is not None else None

        self.get_ready()

    def get_ready(self):
        self.dataset = self.create_distributed_dataset()
        with self.strategy.scope():
            self.segnet = SegmentationNet().build_model(pretrained=self.pretrained_segnet_path,
                                                        input_shape=self.image_h_w_c)

            self.decnet = DecisionNet().build_model(pretrained=self.pretrained_decnet_path,
                                                    mask_shape=self.mask_shape,
                                                    feature_shape=self.feature_shape,
                                                    num_class=self.num_class)

            self.optimizer = build_optimizer(self.parse_config('optimizer'), lr=self.parse_config('lr'))
        self.train_fn = dec_train_fn(self.strategy, self.segnet, self.decnet, self.optimizer)
        self.valid_fn = dec_valid_fn(self.strategy, self.segnet, self.decnet)

    def save_fn(self, epoch):
        self.save_model(epoch, dec=self.decnet)

    def label_fn(self, mask_label):
        size = self.batch_size // self.num_replicas
        if np.sum(mask_label):
            cls_label = 1

        else:
            cls_label = 0

        return tf.constant([cls_label]*size)


class SUTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.weight = self.parse_config('weight')
        self.pretrained_path = os.path.join(self.save_dir, self.parse_config("pretrained_segnet")) if self.parse_config("pretrained_segnet") is not None else None

        self.get_ready()

    def get_ready(self):
        self.dataset = self.create_distributed_dataset()
        with self.strategy.scope():
            self.segnet = u_net(h5_path=self.pretrained_path,
                               input_shape=self.image_h_w_c)
            self.optimizer = build_optimizer(self.parse_config('optimizer'), lr=self.parse_config('lr'))
        self.train_fn = seg_train_fn(self.strategy, self.segnet, self.optimizer, self.weight)
        self.valid_fn = seg_valid_fn(self.strategy, self.segnet, self.weight)

    def save_fn(self, epoch):
        self.save_model(epoch, seg=self.segnet)


class SDTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.weight = self.parse_config('weight')
        self.mask_shape = self.parse_config('mask_shape')
        self.feature_shape = self.parse_config('feature_shape')
        self.num_class = self.parse_config('num_class')

        self.pretrained_segnet_path = os.path.join(self.save_dir, self.parse_config("pretrained_segnet")) if self.parse_config("pretrained_segnet") is not None else None
        self.pretrained_decnet_path = os.path.join(self.save_dir, self.parse_config("pretrained_decnet")) if self.parse_config("pretrained_decnet") is not None else None

        self.get_ready()

    def get_ready(self):
        self.dataset = self.create_distributed_dataset()
        with self.strategy.scope():
            self.segnet = SegmentationNet().build_model(pretrained=self.pretrained_segnet_path,
                                                        input_shape=self.image_h_w_c)

            self.decnet = DecisionNet().build_model(pretrained=self.pretrained_decnet_path,
                                                    mask_shape=self.mask_shape,
                                                    feature_shape=self.feature_shape,
                                                    num_class=self.num_class)

            self.optimizer = build_optimizer(self.parse_config('optimizer'), lr=self.parse_config('lr'))
        self.train_fn = segdec_train_fn(self.strategy, self.segnet, self.decnet,
                                        self.optimizer, self.weight, self.batch_size)
        self.valid_fn = segdec_valid_fn(self.strategy, self.segnet, self.decnet, self.weight, self.batch_size)

    def label_fn(self, mask_label):
        size = self.batch_size // self.num_replicas
        if np.sum(mask_label):
            cls_label = 1

        else:
            cls_label = 0

        return mask_label, tf.constant([cls_label]*size)

    def save_fn(self, epoch):
        self.save_model(epoch, seg=self.segnet, dec=self.decnet)
