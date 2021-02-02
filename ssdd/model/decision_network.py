import tensorflow as tf


class DecisionNet:
    def __init__(self):
        pass

    def build_model(self, pretrained=None, mask_shape=None, feature_shape=None, num_class=None):
        if pretrained is None:
            print('=======================')
            print('new model bulid !')
            print('=======================')
            return self.network(mask_shape=mask_shape, feature_shape=feature_shape, num_class=num_class)

        elif pretrained is not None:
            print('==============================')
            print('pretrained model build !')
            print('==============================')
            return tf.keras.models.load_model(pretrained)

        else:
            print('pretrained type error : {}'.format(pretrained))

            raise None

    @staticmethod
    def network(mask_shape, feature_shape, num_class):
        """
        :param mask_shape: segmentation network's output map -> Tensor Shape (b, h, w, 1)
        :param feature_shape: segmentation network's 15x15 convolution layer output -> Tensor Shape (b, h, w, 1024)
        :param num_class: number of class
        :return: logits -> Tensor Shape(b, num_class)
        """
        try:
            inputs_1 = tf.keras.layers.Input(shape=mask_shape[1:])
            inputs_2 = tf.keras.layers.Input(shape=feature_shape[1:])

            x_1_max_pool = tf.keras.layers.GlobalMaxPool2D()(inputs_1)
            x_1_average_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs_1)

            x_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(inputs_2)

            x_2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), padding='same', name='dec_conv_1')(x_2)
            x_2 = tf.keras.layers.LayerNormalization(axis=[1, 2])(x_2)
            x_2 = tf.keras.layers.Activation('relu')(x_2)
            x_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x_2)

            x_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', name='dec_conv_2')(x_2)
            x_2 = tf.keras.layers.LayerNormalization(axis=[1, 2])(x_2)
            x_2 = tf.keras.layers.Activation('relu')(x_2)
            x_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x_2)

            x_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', name='dec_conv_3')(x_2)
            x_2 = tf.keras.layers.LayerNormalization(axis=[1, 2])(x_2)
            x_2 = tf.keras.layers.Activation('relu')(x_2)

            x_2_max_pool = tf.keras.layers.GlobalMaxPool2D()(x_2)
            x_2_average_pool = tf.keras.layers.GlobalAveragePooling2D()(x_2)

            concat_output = tf.keras.layers.Concatenate(axis=1)([x_2_max_pool,
                                                                 x_2_average_pool,
                                                                 x_1_average_pool,
                                                                 x_1_max_pool])

            fully_conneted = tf.keras.layers.Dense(num_class)(concat_output)

        except TypeError as e:
            print(e)
            print('Please Check mask or feature shape :\n input shape : mask shape = {}, feature shape ={}'.
                  format(mask_shape, feature_shape))
            return None

        except ValueError as e:
            print(e)
            print('Please Check number of class :\n input num_class : {}'. format(num_class))
            return None

        return tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=fully_conneted)
