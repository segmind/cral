import os
from functools import reduce, wraps

import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Add, AveragePooling2D, BatchNormalization,
                                     Conv2D, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Input, LeakyReLU,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_file, get_source_inputs
from tensorflow.python.keras.applications.imagenet_utils import \
    obtain_input_shape


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def preprocess_input(image_array):
    image_array = image_array / 255.
    return image_array


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (
        2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D."""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(
                x)
        x = Add()([x, y])
    return x


def darknet_body(inputs):
    """Darknent body having 52 Convolution2D layers."""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(inputs)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def Darknet53(input_shape=None,
              input_tensor=None,
              include_top=False,
              weights='imagenet',
              pooling=None,
              classes=1000,
              classifier_activation='softmax',
              **kwargs):
    """Generate darknet53 model for Imagenet classification."""

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top`'
            ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = obtain_input_shape(
        input_shape,
        default_size=416,  # multiple of 32 only
        min_size=28,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x = darknet_body(img_input)

    if include_top:
        model_name = 'darknet53'
        x = AveragePooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
        x = Conv2D(
            classes, (1, 1), padding='same', activation=classifier_activation)(
                x)
    else:
        model_name = 'darknet53_headless'
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/transfer-learning/classification/darknet53_weights.h5'  # noqa: E501
        else:
            url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/transfer-learning/classification/darknet53_notop_weights.h5'  # noqa: E501

        weights_file = get_file('darknet53_weights.h5', url)
        file_path = os.path.join(weights_file)
        model.load_weights(file_path)

    elif weights is not None:
        model.load_weights(weights)

    return model
