import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from cral.common import classification_networks
from .utils import LinkNetConfig


def Conv3x3BnReLU(filters, name=None):

    def wrapper(input_tensor):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same', name=name)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    return wrapper


def Conv1x1BnReLU(filters, name=None):

    def wrapper(input_tensor):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same', name=name)(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    return wrapper


def DecoderUpsamplingX2Block(filters, stage):
    conv_block1_name = 'decoder_stage{}a'.format(stage)
    conv_block2_name = 'decoder_stage{}b'.format(stage)
    conv_block3_name = 'decoder_stage{}c'.format(stage)
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    add_name = 'decoder_stage{}_add'.format(stage)

    channels_axis = 3
    # if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        input_filters = K.int_shape(input_tensor)[channels_axis]
        output_filters = K.int_shape(skip)[channels_axis] if skip is not None else filters  # noqa: E501

        x = Conv1x1BnReLU(
            input_filters // 4, name=conv_block1_name)(input_tensor)
        x = tf.keras.layers.UpSampling2D((2, 2), name=up_name)(x)
        x = Conv3x3BnReLU(input_filters // 4, name=conv_block2_name)(x)
        x = Conv1x1BnReLU(output_filters, name=conv_block3_name)(x)

        if skip is not None:
            x = tf.keras.layers.Add(name=add_name)([x, skip])
        return x

    return wrapper


def LinkNet_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv_pw_11_relu').output
    f2 = base_model.get_layer('conv_pw_5_relu').output
    f3 = base_model.get_layer('conv_pw_3_relu').output
    f4 = base_model.get_layer('conv_pw_1_relu').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_mobilenet')
    return model, preprocessing_function


def LinkNet_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block6_out').output
    f2 = base_model.get_layer('conv3_block4_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet50')
    return model, preprocessing_function


def LinkNet_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block23_out').output
    f2 = base_model.get_layer('conv3_block4_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet101')
    return model, preprocessing_function


def LinkNet_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block36_out').output
    f2 = base_model.get_layer('conv3_block8_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet152')
    return model, preprocessing_function


def LinkNet_resnet50v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50v2'](
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block6_1_relu').output
    f2 = base_model.get_layer('conv3_block4_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet50v2')
    return model, preprocessing_function


def LinkNet_resnet101v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101v2'](  # noqa: E501
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block23_1_relu').output
    f2 = base_model.get_layer('conv3_block4_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet101v2')
    return model, preprocessing_function


def LinkNet_resnet152v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152v2'](  # noqa: E501
      input_shape=config.input_shape, weights=weights, include_top=False)
    # img_height = config.height
    # img_width = config.width
    base_model.trainable = base_trainable
    decoder_filters = (None, None, None, None, 16)
    upsample_blocks = 4
    # input = base_model.input
    o = base_model.output

    f1 = base_model.get_layer('conv4_block36_1_relu').output
    f2 = base_model.get_layer('conv3_block8_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    for i in range(upsample_blocks):
        skip = skips[i]
        o = DecoderUpsamplingX2Block(decoder_filters[i], stage=i)(o, skip)

    o = tf.keras.layers.UpSampling2D(size=(2, 2),
                                     interpolation='bilinear',
                                     name='final_upsampling')(o)
    o = tf.keras.layers.Conv2D(filters=num_classes,
                               kernel_size=(3, 3),
                               padding='same',
                               use_bias=True,
                               kernel_initializer='glorot_uniform',
                               name='final_conv')(o)
    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input,
                  outputs=o, name='LinkNet_resnet152v2')
    return model, preprocessing_function


def create_LinkNet(feature_extractor,
                   config,
                   num_classes,
                   weights,
                   base_trainable=True):
    assert isinstance(config, LinkNetConfig), 'please provide a `LinkNetConfig()` object'  # noqa: E501
    if feature_extractor == 'mobilenet':
        return LinkNet_mobilenet(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50':
        return LinkNet_resnet50(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101':
        return LinkNet_resnet101(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152':
        return LinkNet_resnet152(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50v2':
        return LinkNet_resnet50v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101v2':
        return LinkNet_resnet101v2(config, num_classes, weights, base_trainable)  # noqa: E501
    elif feature_extractor == 'resnet152v2':
        return LinkNet_resnet152v2(config, num_classes, weights, base_trainable)  # noqa: E501

    else:
        assert False, "Supported Backbones: 'resnet50', 'resnet101', 'resnet152', 'mobilenet', 'resnet50v2', 'resnet101v2', 'resnet152v2'"  # noqa: E501
