from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Cropping2D, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import RandomNormal

import os


def conv_block(x, num_filter):
    assert tuple(x.shape[1:3]) > (4, 4), 'conv_block 의 input size 는 최소 5 이상이어야 합니다'
    assert num_filter > 0, 'conv block의 num filter 는 최소 1 이상이어야 합니다'

    for _ in range(2):
        x = Conv2D(filters=num_filter,
                   kernel_size=(3, 3),
                   padding='valid',
                   kernel_initializer=RandomNormal(mean=0.))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def skip_connection(x1, x2):
    h1, w1 = x1.shape[1:3]
    h2, w2 = x2.shape[1:3]
    assert h1 >= h2 and w1 >= w2, 'x1 의 size 는 x2 보다 크거나 같아야 합니다'

    h_crop = int((h1 - h2) / 2)
    w_crop = int((w1 - w2) / 2)

    x1 = Cropping2D(cropping=((h_crop, h1-h2-h_crop), (w_crop, w1-w2-w_crop)))(x1)
    x = Concatenate(axis=-1)([x1, x2])

    return x


def skip_connector(x1):
    def _connector(x2):
        x = skip_connection(x1, x2)

        return x
    return _connector


def down_sampling(x):
    num_filter = int(x.shape[-1] * 2)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = conv_block(x, num_filter)

    return x


def up_sampling(x, connector):
    num_filter = int(x.shape[-1] / 2)
    x = Conv2DTranspose(filters=num_filter,
                        kernel_size=(3, 3),
                        strides=2,
                        padding='same',
                        kernel_initializer=RandomNormal(mean=0.))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = connector(x)
    x = conv_block(x, num_filter)

    return x


def u_net(h5_path=None, input_shape=None, num_down_sampling=3, num_output_filter=1):
    if h5_path is not None and os.path.exists(h5_path):
        print('build pretrained model')

        return load_model(h5_path)
    elif input_shape is not None:
        print('build new model')
        x = inputs = Input(input_shape)
        connectors = []
        x = conv_block(x, 64)
        for _conv in [down_sampling] * num_down_sampling:
            connectors.append(skip_connector(x))
            x = _conv(x)

        for _conn in reversed(connectors):
            x = up_sampling(x, _conn)

        outputs = Conv2D(filters=num_output_filter,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer=RandomNormal(mean=0.))(x)

        return Model(inputs=inputs, outputs=[x, outputs])
    else:
        print(' * Model Build 실패.')
        print(' * h5_path : {}'.format(h5_path))
        print(' * input_shape : {}'.format(input_shape))

        return None
