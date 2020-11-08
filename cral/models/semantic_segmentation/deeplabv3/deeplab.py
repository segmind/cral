from cral.common import classification_networks
from cral.models.semantic_segmentation.deeplabv3.deeplab_xception import \
    DeepLabV3Plus_xception
from cral.models.semantic_segmentation.deeplabv3.deeplabv3_mobilenet import \
    DeepLabV3Plus_mobilenet
from cral.models.semantic_segmentation.deeplabv3.utils import (Deeplabv3Config,
                                                               Upsample)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, concatenate)
from tensorflow.keras.models import Model


def ASPP(tensor, atrous_rates):
    """atrous spatial pyramid pooling."""
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(
        pool_size=(dims[1], dims[2]), name='average_pooling')(
            tensor)
    y_pool = Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal',
        name='pool_1x1conv2d',
        use_bias=False)(
            y_pool)
    y_pool = BatchNormalization(name='bn_1')(y_pool)
    y_pool = Activation('relu', name='relu_1')(y_pool)

    y_pool = Upsample(
        height=dims[1],
        width=dims[2],
        name=y_pool.name.split('/')[0] + '_upsample')(
            y_pool)

    y_1 = Conv2D(
        filters=256,
        kernel_size=1,
        dilation_rate=1,
        padding='same',
        kernel_initializer='he_normal',
        name='ASPP_conv2d_d1',
        use_bias=False)(
            tensor)
    y_1 = BatchNormalization(name='bn_2')(y_1)
    y_1 = Activation('relu', name='relu_2')(y_1)

    y_6 = Conv2D(
        filters=256,
        kernel_size=3,
        dilation_rate=atrous_rates[0],
        padding='same',
        kernel_initializer='he_normal',
        name='ASPP_conv2d_d6',
        use_bias=False)(
            tensor)
    y_6 = BatchNormalization(name='bn_3')(y_6)
    y_6 = Activation('relu', name='relu_3')(y_6)

    y_12 = Conv2D(
        filters=256,
        kernel_size=3,
        dilation_rate=atrous_rates[1],
        padding='same',
        kernel_initializer='he_normal',
        name='ASPP_conv2d_d12',
        use_bias=False)(
            tensor)
    y_12 = BatchNormalization(name='bn_4')(y_12)
    y_12 = Activation('relu', name='relu_4')(y_12)

    y_18 = Conv2D(
        filters=256,
        kernel_size=3,
        dilation_rate=atrous_rates[2],
        padding='same',
        kernel_initializer='he_normal',
        name='ASPP_conv2d_d18',
        use_bias=False)(
            tensor)
    y_18 = BatchNormalization(name='bn_5')(y_18)
    y_18 = Activation('relu', name='relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(
        filters=256,
        kernel_size=1,
        dilation_rate=1,
        padding='same',
        kernel_initializer='he_normal',
        name='ASPP_conv2d_final',
        use_bias=False)(
            y)
    y = BatchNormalization(name='bn_final')(y)
    y = Activation('relu', name='relu_final')(y)
    return y


def DeepLabV3Plus_resnet50(config, num_classes, weights, base_trainable):

    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    img_height = config.height
    img_width = config.width

    base_model.trainable = base_trainable

    image_features = base_model.get_layer('conv4_block6_out').output

    x_a = ASPP(image_features, config.atrous_rates)
    x_a = Upsample(height=img_height // 4, width=img_width // 4)(x_a)

    # x_b = base_model.get_layer('activation_9').output
    x_b = base_model.get_layer('conv2_block3_out').output
    x_b = Conv2D(
        filters=48,
        kernel_size=1,
        padding='same',
        kernel_initializer='he_normal',
        name='low_level_projection',
        use_bias=False)(
            x_b)
    x_b = BatchNormalization(name='bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    x = concatenate([x_a, x_b], name='decoder_concat')

    x = Conv2D(
        filters=256,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='decoder_conv2d_1',
        use_bias=False)(
            x)
    x = BatchNormalization(name='bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(
        filters=256,
        kernel_size=3,
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        name='decoder_conv2d_2',
        use_bias=False)(
            x)
    x = BatchNormalization(name='bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)
    x = Upsample(height=img_height, width=img_width)(x)

    x = Conv2D(num_classes, (1, 1), name='output_layer')(x)

    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    # print(f'*** Output_Shape => {model.output_shape} ***')
    return model, preprocessing_function


def create_DeepLabv3Plus(feature_extractor,
                         config,
                         num_classes,
                         weights,
                         base_trainable=True):

    assert isinstance(
        config, Deeplabv3Config), 'please provide a `Deeplabv3Config()` object'

    if feature_extractor == 'resnet50':
        return DeepLabV3Plus_resnet50(config, num_classes, weights,
                                      base_trainable)
    elif feature_extractor == 'xception':
        return DeepLabV3Plus_xception(config, num_classes, weights,
                                      base_trainable)
    elif feature_extractor == 'mobilenetv2':
        return DeepLabV3Plus_mobilenet(config, num_classes, weights,
                                       base_trainable)
    else:
        assert False, 'only resnet50, xception and mobilenetv2 are supported'
