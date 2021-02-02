import tensorflow as tf


def build_optimizer(name="sgd", **kwargs):
    if name == "sgd":
        lr = 0.01
        momentum = 0.0
        if 'lr' in kwargs:
            lr = kwargs['lr']
        if 'momentum' in kwargs:
            momentum = kwargs['momentum']

        return tf.keras.optimizers.SGD(lr=lr, momentum=momentum)

    if name == 'adam':
        lr = 0.01

        if 'lr' in kwargs:
            lr = kwargs['lr']

        return tf.keras.optimizers.Adam(lr=lr)
