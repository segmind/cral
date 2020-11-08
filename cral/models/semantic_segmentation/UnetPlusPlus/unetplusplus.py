from cral.common import classification_networks
from cral.models.semantic_segmentation.UnetPlusPlus.utils import (
    UnetPlusPlusConfig, Upsample, standard_unit)
from tensorflow.keras.layers import (Average, Conv2D, Conv2DTranspose,
                                     concatenate)
from tensorflow.keras.models import Model

bn_axis = 3


def ret_model_output(config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1):
    nb_filter = config.filters

    up1_2 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(
            conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    up2_2 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(
            conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])
    up1_3 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(
            conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2],
                          name='merge13',
                          axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    up3_2 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(
            conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(
            conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2],
                          name='merge23',
                          axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(
            conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3],
                          name='merge14',
                          axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    up4_2 = Conv2DTranspose(
        nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(
            conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(
            conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2],
                          name='merge33',
                          axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(
            conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3],
                          name='merge24',
                          axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(
            conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4],
                          name='merge15',
                          axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    return conv1_2, conv1_3, conv1_4, conv1_5


def get_output_list(config, num_classes, conv1_2, conv1_3, conv1_4, conv1_5):
    nestnet_output_1 = Conv2D(
        num_classes, (3, 3), name='output_1', padding='same')(
            conv1_2)
    nestnet_output_2 = Conv2D(
        num_classes, (3, 3), name='output_2', padding='same')(
            conv1_3)
    nestnet_output_3 = Conv2D(
        num_classes, (3, 3), name='output_3', padding='same')(
            conv1_4)
    nestnet_output_4 = Conv2D(
        num_classes, (3, 3), name='output_4', padding='same')(
            conv1_5)
    output_list = []
    if (config.deep_supervision is True):
        if (config.num_upsample_layers == 5):
            output_list = [
                nestnet_output_1, nestnet_output_2, nestnet_output_3,
                nestnet_output_4
            ]
        elif (config.num_upsample_layers == 4):
            output_list = [
                nestnet_output_1, nestnet_output_2, nestnet_output_3
            ]
        elif (config.num_upsample_layers == 3):
            output_list = [nestnet_output_1, nestnet_output_2]
        else:
            output_list = [nestnet_output_1]
    else:
        if (config.num_upsample_layers == 5):
            output_list = [nestnet_output_4]
        elif (config.num_upsample_layers == 4):
            output_list = [nestnet_output_3]
        elif (config.num_upsample_layers == 3):
            output_list = [nestnet_output_2]
        else:
            output_list = [nestnet_output_1]
    return output_list


def UnetPlusPlus_vgg16(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg16'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('block1_pool').output
    conv2_1 = base_model.get_layer('block2_pool').output
    conv3_1 = base_model.get_layer('block3_pool').output
    conv4_1 = base_model.get_layer('block4_pool').output
    conv5_1 = base_model.get_layer('block5_pool').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_vgg16')
    return model, preprocessing_function


def UnetPlusPlus_vgg19(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg19'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('block1_pool').output
    conv2_1 = base_model.get_layer('block2_pool').output
    conv3_1 = base_model.get_layer('block3_pool').output
    conv4_1 = base_model.get_layer('block4_pool').output
    conv5_1 = base_model.get_layer('block5_pool').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_vgg19')
    return model, preprocessing_function


def UnetPlusPlus_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_relu').output
    conv2_1 = base_model.get_layer('conv2_block3_out').output
    conv3_1 = base_model.get_layer('conv3_block4_out').output
    conv4_1 = base_model.get_layer('conv4_block6_out').output
    conv5_1 = base_model.get_layer('conv5_block3_out').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet50')
    return model, preprocessing_function


def UnetPlusPlus_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_relu').output
    conv2_1 = base_model.get_layer('conv2_block3_out').output
    conv3_1 = base_model.get_layer('conv3_block4_out').output
    conv4_1 = base_model.get_layer('conv4_block23_out').output
    conv5_1 = base_model.get_layer('conv5_block3_out').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet101')
    return model, preprocessing_function


def UnetPlusPlus_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_relu').output
    conv2_1 = base_model.get_layer('conv2_block3_out').output
    conv3_1 = base_model.get_layer('conv3_block8_out').output
    conv4_1 = base_model.get_layer('conv4_block36_out').output
    conv5_1 = base_model.get_layer('conv5_block3_out').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet152')
    return model, preprocessing_function


def UnetPlusPlus_resnet50v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50v2'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_conv').output
    conv2_1 = base_model.get_layer('conv2_block3_1_relu').output
    conv3_1 = base_model.get_layer('conv3_block4_1_relu').output
    conv4_1 = base_model.get_layer('conv4_block6_1_relu').output
    conv5_1 = base_model.get_layer('post_relu').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet50v2')
    return model, preprocessing_function


def UnetPlusPlus_resnet101v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet101v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_conv').output
    conv2_1 = base_model.get_layer('conv2_block3_1_relu').output
    conv3_1 = base_model.get_layer('conv3_block4_1_relu').output
    conv4_1 = base_model.get_layer('conv4_block23_1_relu').output
    conv5_1 = base_model.get_layer('post_relu').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet101v2')
    return model, preprocessing_function


def UnetPlusPlus_resnet152v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet152v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv1_conv').output
    conv2_1 = base_model.get_layer('conv2_block3_1_relu').output
    conv3_1 = base_model.get_layer('conv3_block8_1_relu').output
    conv4_1 = base_model.get_layer('conv4_block36_1_relu').output
    conv5_1 = base_model.get_layer('post_relu').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_resnet152v2')
    return model, preprocessing_function


def UnetPlusPlus_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
        input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('conv_pw_1_relu').output
    conv3_1 = base_model.get_layer('conv_pw_5_relu').output
    conv2_1 = base_model.get_layer('conv_pw_3_relu').output
    conv4_1 = base_model.get_layer('conv_pw_11_relu').output
    conv5_1 = base_model.get_layer('conv_pw_13_relu').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_mobilenet')
    return model, preprocessing_function


def UnetPlusPlus_mobilenetv2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'mobilenetv2'](
            input_shape=config.input_shape, weights=weights, include_top=False)
    base_model.trainable = base_trainable
    img_height = config.height
    img_width = config.width
    conv1_1 = base_model.get_layer('block_1_expand_relu').output
    conv2_1 = base_model.get_layer('block_3_expand_relu').output
    conv3_1 = base_model.get_layer('block_6_expand_relu').output
    conv4_1 = base_model.get_layer('block_13_expand_relu').output
    conv5_1 = base_model.get_layer('out_relu').output
    conv1_2, conv1_3, conv1_4, conv1_5 = ret_model_output(
        config, conv1_1, conv2_1, conv3_1, conv4_1, conv5_1)
    output_list = get_output_list(config, num_classes, conv1_2, conv1_3,
                                  conv1_4, conv1_5)
    o = 1
    if (len(output_list) > 1):
        o = Average()(output_list)
    else:
        o = output_list[0]
    o = Upsample(img_height, img_width)(o)
    model = Model(
        inputs=base_model.input, outputs=o, name='UnetPlusPlus_mobilenetv2')
    return model, preprocessing_function


def create_UnetPlusPlus(feature_extractor,
                        config,
                        num_classes,
                        weights,
                        base_trainable=True):

    assert isinstance(
        config,
        UnetPlusPlusConfig), 'please provide a `UnetPlusPlusConfig()` object'
    if feature_extractor == 'vgg16':
        return UnetPlusPlus_vgg16(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg19':
        return UnetPlusPlus_vgg19(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50':
        return UnetPlusPlus_resnet50(config, num_classes, weights,
                                     base_trainable)
    elif feature_extractor == 'resnet101':
        return UnetPlusPlus_resnet101(config, num_classes, weights,
                                      base_trainable)
    elif feature_extractor == 'resnet152':
        return UnetPlusPlus_resnet152(config, num_classes, weights,
                                      base_trainable)
    elif feature_extractor == 'resnet50v2':
        return UnetPlusPlus_resnet50v2(config, num_classes, weights,
                                       base_trainable)
    elif feature_extractor == 'resnet101v2':
        return UnetPlusPlus_resnet101v2(config, num_classes, weights,
                                        base_trainable)
    elif feature_extractor == 'resnet152v2':
        return UnetPlusPlus_resnet152v2(config, num_classes, weights,
                                        base_trainable)
    elif feature_extractor == 'mobilenet':
        return UnetPlusPlus_mobilenet(config, num_classes, weights,
                                      base_trainable)
    elif feature_extractor == 'mobilenetv2':
        return UnetPlusPlus_mobilenetv2(config, num_classes, weights,
                                        base_trainable)
    else:
        assert False, 'Supported Backbones -> [vgg16, vgg19, resnet50, resnet50v2, resnet101, resnet101v2, resnet152, resnet152v2, mobilenet, mobilenetv2]'  # noqa: E501
