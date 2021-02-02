import tensorflow as tf


def segmentation_loss(mask_labels, mask_outputs, weight):
    x = mask_outputs
    z = mask_labels

    weight_masks = z * weight + tf.cast(tf.math.logical_not(tf.cast(z, tf.bool)), tf.float32)
    # sigmoid_cross_entropy_with_logits 수식
    loss = tf.math.maximum(x, 0) - x * z + tf.math.log(1 + tf.math.exp(-tf.math.abs(x)))
    loss *= weight_masks
    loss = tf.math.reduce_mean(loss, axis=(1, 2))

    return tf.math.reduce_sum(loss)


def classification_loss(y_true, y_pred):
    """

    :param y_true: tensor of shape (batch size)
                    0 = negative label
                    1 = positive label

    :param y_pred: tensor of shape (batch size, num class)

    :return: cross entropy loss
    """
    y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)

    return tf.math.reduce_sum(loss)
