from cral.common import classification_networks
from tensorflow.keras.layers import (BatchNormalization, Conv2D, UpSampling2D,
                                     ZeroPadding2D)
from tensorflow.keras.models import Model

from .utils import SegNetConfig


def ret_model_output(config, f2, f3, f4, f5):
    o = f3
    if (config.num_upsample_layers == 2):
        o = f2
    elif (config.num_upsample_layers == 4):
        o = f4
    elif (config.num_upsample_layers == 5):
        o = f5
    else:
        o = f3
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    for _ in range(config.num_upsample_layers - 2):
        o = (UpSampling2D((2, 2)))(o)
        o = (ZeroPadding2D((1, 1)))(o)
        o = (Conv2D(128, (3, 3), padding='valid'))(o)
        o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid'))(o)
    o = (BatchNormalization())(o)
    return o


def SegNet_vgg16(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg16'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block2_pool').output
    f3 = base_model.get_layer('block3_pool').output
    f4 = base_model.get_layer('block4_pool').output
    f5 = base_model.get_layer('block5_pool').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_vgg16')
    return model, preprocessing_function


def SegNet_vgg19(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg19'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block2_pool').output
    f3 = base_model.get_layer('block3_pool').output
    f4 = base_model.get_layer('block4_pool').output
    f5 = base_model.get_layer('block5_pool').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_vgg19')
    return model, preprocessing_function


def SegNet_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block6_out').output
    f5 = base_model.get_layer('conv5_block3_out').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_resnet50')
    return model, preprocessing_function


def SegNet_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block23_out').output
    f5 = base_model.get_layer('conv5_block3_out').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_resnet101')
    return model, preprocessing_function


def SegNet_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block8_out').output
    f4 = base_model.get_layer('conv4_block36_out').output
    f5 = base_model.get_layer('conv5_block3_out').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_resnet152')
    return model, preprocessing_function


def SegNet_resnet50v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50v2'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block4_1_relu').output
    f4 = base_model.get_layer('conv4_block6_1_relu').output
    f5 = base_model.get_layer('post_relu').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_resnet50v2')
    return model, preprocessing_function


def SegNet_resnet101v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet101v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block4_1_relu').output
    f4 = base_model.get_layer('conv4_block23_1_relu').output
    f5 = base_model.get_layer('post_relu').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='SegNet_resnet101v2')
    return model, preprocessing_function


def SegNet_resnet152v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet152v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv2_block3_1_relu').output
    f3 = base_model.get_layer('conv3_block8_1_relu').output
    f4 = base_model.get_layer('conv4_block36_1_relu').output
    f5 = base_model.get_layer('post_relu').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='SegNet_resnet152v2')
    return model, preprocessing_function


def SegNet_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('conv_pw_3_relu').output
    f3 = base_model.get_layer('conv_pw_5_relu').output
    f4 = base_model.get_layer('conv_pw_11_relu').output
    f5 = base_model.get_layer('conv_pw_13_relu').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(inputs=base_model.input, outputs=o, name='SegNet_mobilenet')
    return model, preprocessing_function


def SegNet_mobilenetv2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'mobilenetv2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    f2 = base_model.get_layer('block_3_expand_relu').output
    f3 = base_model.get_layer('block_6_expand_relu').output
    f4 = base_model.get_layer('block_13_expand_relu').output
    f5 = base_model.get_layer('out_relu').output
    o = ret_model_output(config, f2, f3, f4, f5)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='SegNet_mobilenetv2')
    return model, preprocessing_function


def create_SegNet(feature_extractor,
                  config,
                  num_classes,
                  weights,
                  base_trainable=True):

    assert isinstance(config,
                      SegNetConfig), 'please provide a `SegNetConfig()` object'
    if feature_extractor == 'vgg16':
        return SegNet_vgg16(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg19':
        return SegNet_vgg19(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50':
        return SegNet_resnet50(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101':
        return SegNet_resnet101(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152':
        return SegNet_resnet152(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50v2':
        return SegNet_resnet50v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101v2':
        return SegNet_resnet101v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152v2':
        return SegNet_resnet152v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenet':
        return SegNet_mobilenet(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenetv2':
        return SegNet_mobilenetv2(config, num_classes, weights, base_trainable)
    else:
        assert False, 'Supported Backbones -> [resnet50, resnet101, resnet152,\
        resnet50v2, resnet101v2, resnet152v2, vgg16, vgg19, mobilenet,\
        mobilenetv2]'
