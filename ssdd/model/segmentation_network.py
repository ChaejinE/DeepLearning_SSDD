import tensorflow as tf


class SegmentationNet:
    def __init__(self):
        pass

    def build_model(self, pretrained=None, input_shape=None):
        if pretrained is None:
            print('=======================')
            print('new model bulid !')
            print('=======================')
            return self.network(input_shape)

        elif pretrained is not None:
            print('==============================')
            print('pretrained model build !')
            print('==============================')
            return tf.keras.models.load_model(pretrained)

        else:
            print('pretrained type error : {}'.format(pretrained))

            raise None

    @staticmethod
    def network(image_shape):
        """
        :param image_shape: input image's shape
        :return: [x, segmentation_output] -> [1024 channel feature map, segmentation mask map]
        """
        try:
            inputs = tf.keras.layers.Input(shape=image_shape)
            x = tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=32,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)

            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Conv2D(filters=64,
                                       kernel_size=(5, 5),
                                       padding='same',
                                       kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.))(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

            x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(15, 15), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            segmentation_output = tf.keras.layers.Conv2D(filters=1,
                                                         kernel_size=(1, 1),
                                                         padding='same')(x)
        except ValueError as e:
            print(e)
            print('Please Check image shape :\n input shape : image shape = {}'.format(image_shape))
            raise ValueError

        return tf.keras.Model(inputs=inputs, outputs=[x, segmentation_output])

