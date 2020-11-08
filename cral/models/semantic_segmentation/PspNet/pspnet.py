from cral.common import classification_networks
from cral.models.semantic_segmentation.PspNet.utils import (PspNetConfig,
                                                            Upsample,
                                                            pool_block)
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D)
from tensorflow.keras.models import Model

pool_factors = [1, 2, 3, 6]


def PspNet_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block6_out').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_resnet50')
    return model, preprocessing_function


def PspNet_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block23_out').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_resnet101')
    return model, preprocessing_function


def PspNet_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block8_out').output
    f4 = base_model.get_layer('conv4_block36_out').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_resnet152')
    return model, preprocessing_function


def PspNet_vgg16(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg16'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block3_conv3').output
    f3 = base_model.get_layer('block4_conv3').output
    f4 = base_model.get_layer('block5_conv3').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_vgg16')
    return model, preprocessing_function


def PspNet_vgg19(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg19'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block3_conv4').output
    f3 = base_model.get_layer('block4_conv4').output
    f4 = base_model.get_layer('block5_conv4').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_vgg19')
    return model, preprocessing_function


def PspNet_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv_pw_3_relu').output
    f3 = base_model.get_layer('conv_pw_5_relu').output
    f4 = base_model.get_layer('conv_pw_11_relu').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_mobilenet')
    return model, preprocessing_function


def PspNet_mobilenetv2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'mobilenetv2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block_3_expand_relu').output
    f3 = base_model.get_layer('block_6_expand_relu').output
    f4 = base_model.get_layer('block_13_expand_relu').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='PspNet_mobilenetv2')
    return model, preprocessing_function


def PspNet_resnet50v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50v2'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block4_1_relu').output
    f4 = base_model.get_layer('conv4_block6_1_relu').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(inputs=base_model.input, outputs=o, name='PspNet_resnet50v2')
    return model, preprocessing_function


def PspNet_resnet101v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet101v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block4_1_relu').output
    f4 = base_model.get_layer('conv4_block23_1_relu').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='PspNet_resnet101v2')
    return model, preprocessing_function


def PspNet_resnet152v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet152v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    img_height = config.height
    img_width = config.width
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block8_1_relu').output
    f4 = base_model.get_layer('conv4_block36_1_relu').output
    o = f3
    if (config.down_sample_factor == 4):
        o = f2
    elif (config.down_sample_factor == 16):
        o = f4
    else:
        o = f3
    pool_outs = [o]
    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)
    o = Concatenate(axis=3)(pool_outs)
    o = Conv2D(512, (1, 1), use_bias=False)(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='PspNet_resnet152v2')
    return model, preprocessing_function


def create_PspNet(feature_extractor,
                  config,
                  num_classes,
                  weights,
                  base_trainable=True):

    assert isinstance(config,
                      PspNetConfig), 'please provide a `PspNetConfig()` object'

    if feature_extractor == 'resnet50':
        return PspNet_resnet50(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50v2':
        return PspNet_resnet50v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101v2':
        return PspNet_resnet101v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152v2':
        return PspNet_resnet152v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101':
        return PspNet_resnet101(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152':
        return PspNet_resnet152(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg16':
        return PspNet_vgg16(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg19':
        return PspNet_vgg19(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenet':
        return PspNet_mobilenet(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenetv2':
        return PspNet_mobilenetv2(config, num_classes, weights, base_trainable)
    else:
        assert False, 'Supported Backbones -> [resnet50, resnet101, resnet152, \
        resnet50v2, resnet101v2, resnet152v2, vgg16, vgg19,\
        mobilenet, mobilenetv2]'
