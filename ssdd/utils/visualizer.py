import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
import sys


class Visualizer:
    def __init__(self, segnet_model, decnet_model, layer_name=None):
        self.segnet = segnet_model
        self.decnet = decnet_model

        self.layer_name = self.find_target_layer() if layer_name is None else layer_name

        self.grad_model = tf.keras.models.Model(
            inputs=[self.decnet.inputs],
            outputs=[self.decnet.get_layer(self.layer_name).output,
                     self.decnet.output])

    def find_target_layer(self):
        for layer in reversed(self.decnet.layers):
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def grad_cam(self, image, class_idx=0, eps=1e-8):
        image = tf.cast(image, tf.float32)
        feature_maps, segmentation_mask = self.segnet(image)

        with tf.GradientTape() as tape:
            (convOutputs, predictions) = self.grad_model([segmentation_mask, feature_maps])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        segmentation_mask = segmentation_mask[0] * 255
        segmentation_mask = np.array(segmentation_mask, dtype=np.float32)
        segmentation_mask = cv2.resize(segmentation_mask, (w, h))

        return heatmap, segmentation_mask, np.argmax(predictions, axis=-1)[0]

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)

    def save_result_image(self, src_folder, dst_folder, file_fmt='.jpg', class_idx=1):
        names = sorted([name for name in os.listdir(src_folder) if file_fmt in name])
        for i, name in enumerate(names):
            sys.stdout.write("\r {} / {}".format(i+1, len(names)))
            sys.stdout.flush()

            image = cv2.imread(os.path.join(src_folder, name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256, 704))
            _image = np.expand_dims(image, axis=(0, -1))
            _image = np.array(_image, dtype=np.float32) / 255.

            heat_map, mask, pred = self.grad_cam(_image, class_idx=class_idx)
            save_image = np.concatenate([image, heat_map, mask], axis=1)

            cv2.imwrite(os.path.join(dst_folder, '{}_{}_{}_pred_{}.jpg'.format(name.split('.')[0], i, class_idx, pred)), save_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segnet-h5-path', required=True)
    parser.add_argument('--decnet-h5-path', required=True)
    parser.add_argument('--dataset-folder', required=True)
    parser.add_argument('--save-folder', required=True)
    parser.add_argument('--file-fmt', default='.jpg')
    parser.add_argument('--class-idx', default=1)
    args = parser.parse_args()

    print('build_model...')
    segnet_model = tf.keras.models.load_model(args.segnet_h5_path)
    decnet_model = tf.keras.models.load_model(args.decnet_h5_path)

    print('create visualizer..')
    visualizer = Visualizer(segnet_model, decnet_model)
    visualizer.grad_model.summary()

    print('visualize layer name : {}'.format(visualizer.layer_name))

    visualizer.save_result_image(args.dataset_folder, args.save_folder, file_fmt=args.file_fmt, class_idx=args.class_idx)

    """
    # 실행 코드 샘플    
    python3 ./tf_2/object_detection/defect_detection/model/visualizer.py \
    --segnet-h5-path /Users/hyeseong/virtualenv/yodj_models/yodj_models/tf_2/object_detection/defect_detection/model/segnet_0728_v4_gpu1_epoch_60.h5\
    --decnet-h5-path /Users/hyeseong/virtualenv/yodj_models/yodj_models/tf_2/object_detection/defect_detection/model/decnet_0728_v4_epoch_36.h5\
    --dataset-folder /Users/hyeseong/datasets/private/lens_tube/lens_tube_0908_2nd/splited_roi_0908/pos_0908_valid\
    --save-folder /Users/hyeseong/deep_learning/result_images/temp
    """
