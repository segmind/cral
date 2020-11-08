import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as preprocessing_fn
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, DepthwiseConv2D, Dropout,
                                     GlobalAveragePooling2D, Input, ReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file

from .utils import Expand_Dims, Upsample

WEIGHTS_PATH_MOBILE = 'https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5'  # noqa: E501


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs,
                        expansion,
                        stride,
                        alpha,
                        filters,
                        block_id,
                        skip_connection,
                        rate=1):
    # print('haha')
    # print(inputs)
    in_channels = inputs.shape[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'expand')(
                x)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(
                x)
        x = ReLU(max_value=6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same',
        dilation_rate=(rate, rate),
        name=prefix + 'depthwise')(
            x)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(
            x)

    x = ReLU(max_value=6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'project')(
            x)
    x = BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(
            x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


# def DeepLabV3Plus_xception(config, num_classes, weights, base_trainable):
def DeepLabV3Plus_mobilenet(config,
                            num_classes,
                            weights,
                            base_trainable,
                            alpha=1.):
    input_shape = config.input_shape
    img_input = Input(shape=input_shape)

    if config.output_stride != 8:
        raise ValueError(
            f'only output_stride=8 is supported, given {config.output_stride}')

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(
        first_block_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(
            img_input)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = ReLU(max_value=6, name='Conv_Relu6')(x)

    x = _inverted_res_block(
        x,
        filters=16,
        alpha=alpha,
        stride=1,
        expansion=1,
        block_id=0,
        skip_connection=False)

    x = _inverted_res_block(
        x,
        filters=24,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=1,
        skip_connection=False)
    x = _inverted_res_block(
        x,
        filters=24,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=2,
        skip_connection=True)

    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=2,
        expansion=6,
        block_id=3,
        skip_connection=False)
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=4,
        skip_connection=True)
    x = _inverted_res_block(
        x,
        filters=32,
        alpha=alpha,
        stride=1,
        expansion=6,
        block_id=5,
        skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,  # 1!
        expansion=6,
        block_id=6,
        skip_connection=False)
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=7,
        skip_connection=True)
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=8,
        skip_connection=True)
    x = _inverted_res_block(
        x,
        filters=64,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=9,
        skip_connection=True)

    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=10,
        skip_connection=False)
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=11,
        skip_connection=True)
    x = _inverted_res_block(
        x,
        filters=96,
        alpha=alpha,
        stride=1,
        rate=2,
        expansion=6,
        block_id=12,
        skip_connection=True)

    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=1,
        rate=2,  # 1!
        expansion=6,
        block_id=13,
        skip_connection=False)
    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=1,
        rate=4,
        expansion=6,
        block_id=14,
        skip_connection=True)
    x = _inverted_res_block(
        x,
        filters=160,
        alpha=alpha,
        stride=1,
        rate=4,
        expansion=6,
        block_id=15,
        skip_connection=True)

    x = _inverted_res_block(
        x,
        filters=320,
        alpha=alpha,
        stride=1,
        rate=4,
        expansion=6,
        block_id=16,
        skip_connection=False)

    # shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Expand_Dims()(b4)
    b4 = Expand_Dims()(b4)

    b4 = Conv2D(
        256, (1, 1), padding='same', use_bias=False, name='image_pooling')(
            b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = ReLU(max_value=6)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = Upsample(size_before[1], size_before[2])(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = ReLU(max_value=6, name='aspp0_activation')(b0)

    x = Concatenate()([b4, b0])

    x = Conv2D(
        256, (1, 1), padding='same', use_bias=False, name='concat_projection')(
            x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = ReLU(max_value=6)(x)
    x = Dropout(0.1)(x)

    last_layer_name = 'custom_logits_semantic'

    x = Conv2D(num_classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Upsample(size_before3[1], size_before3[2])(x)

    model = Model(img_input, x, name='deeplabv3plus')

    if weights in ('imagenet', 'pascal_voc'):
        if weights == 'imagenet':
            print('`imagenet` weights are not available, loading `pascal_voc`\
                instead')

        weights_path = get_file(
            'deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH_MOBILE)

    else:
        weights_path = weights

    # print(input_shape)
    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)

    return model, preprocessing_fn
