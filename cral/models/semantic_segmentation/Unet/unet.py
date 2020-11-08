from cral.common import classification_networks
from cral.models.semantic_segmentation.Unet.utils import UNetConfig
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     UpSampling2D, ZeroPadding2D, concatenate)
from tensorflow.keras.models import Model


def UNet_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('conv_pw_1_relu').output
    f2 = base_model.get_layer('conv_pw_3_relu').output
    f3 = base_model.get_layer('conv_pw_5_relu').output
    f4 = base_model.get_layer('conv_pw_11_relu').output
    f5 = base_model.get_layer('conv_pw_13_relu').output
    o = f5

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    # o = (ZeroPadding2D((1, 1)))(o)
    # o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    # o = (BatchNormalization())(o)
    # o = (UpSampling2D((2, 2)))(o)
    # o = (concatenate([o, f1]))

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)
    model = Model(inputs=base_model.input, outputs=o, name='UNet_mobilenet')
    return model, preprocessing_function


def UNet_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('conv1_conv').output
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block6_out').output
    f5 = base_model.get_layer('conv5_block3_out').output

    o = f5

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    # o = (ZeroPadding2D((1, 1)))(o)
    # o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    # o = (BatchNormalization())(o)
    # o = (UpSampling2D((2, 2)))(o)
    # o = (concatenate([o, f1]))

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)

    model = Model(inputs=base_model.input, outputs=o, name='UNet_resnet50')
    return model, preprocessing_function


def UNet_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('conv1_relu').output
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block4_out').output
    f4 = base_model.get_layer('conv4_block23_out').output
    f5 = base_model.get_layer('conv5_block3_out').output

    o = f5

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    # o = (ZeroPadding2D((1, 1)))(o)
    # o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    # o = (BatchNormalization())(o)
    # o = (UpSampling2D((2, 2)))(o)
    # o = (concatenate([o, f1]))

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)

    model = Model(inputs=base_model.input, outputs=o, name='UNet_resnet101')
    return model, preprocessing_function


def UNet_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('conv1_relu').output
    f2 = base_model.get_layer('conv2_block3_out').output
    f3 = base_model.get_layer('conv3_block8_out').output
    f4 = base_model.get_layer('conv4_block36_out').output
    f5 = base_model.get_layer('conv5_block3_out').output

    o = f5

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    # o = (ZeroPadding2D((1, 1)))(o)
    # o = (Conv2D(128, (3, 3), padding='valid' , activation='relu'))(o)
    # o = (BatchNormalization())(o)
    # o = (UpSampling2D((2, 2)))(o)
    # o = (concatenate([o, f1]))

    o = (UpSampling2D((2, 2)))(o)
    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)

    model = Model(inputs=base_model.input, outputs=o, name='UNet_resnet152')
    return model, preprocessing_function


def UNet_vgg16(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg16'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('block1_conv2').output
    f2 = base_model.get_layer('block2_conv2').output
    f3 = base_model.get_layer('block3_conv3').output
    f4 = base_model.get_layer('block4_conv3').output
    f5 = base_model.get_layer('block5_conv3').output
    f6 = base_model.get_layer('block5_pool').output

    o = f6

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f5]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)

    model = Model(inputs=base_model.input, outputs=o, name='UNet_vgg16')
    return model, preprocessing_function


def UNet_vgg19(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg19'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    f1 = base_model.get_layer('block1_conv2').output
    f2 = base_model.get_layer('block2_conv2').output
    f3 = base_model.get_layer('block3_conv4').output
    f4 = base_model.get_layer('block4_conv4').output
    f5 = base_model.get_layer('block5_conv4').output
    f6 = base_model.get_layer('block5_pool').output

    o = f6

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f5]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f4]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f3]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f2]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = (UpSampling2D((2, 2)))(o)
    o = (concatenate([o, f1]))

    o = (ZeroPadding2D((1, 1)))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (BatchNormalization())(o)
    o = Conv2D(num_classes, (3, 3), padding='same')(o)

    o = (Activation('softmax'))(o)

    model = Model(inputs=base_model.input, outputs=o, name='UNet_vgg19')
    return model, preprocessing_function


def create_UNet(feature_extractor,
                config,
                num_classes,
                weights,
                base_trainable=True):
    assert isinstance(config,
                      UNetConfig), 'please provide a `UNetConfig()` object'
    if feature_extractor == 'mobilenet':
        return UNet_mobilenet(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50':
        return UNet_resnet50(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101':
        return UNet_resnet101(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152':
        return UNet_resnet152(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg16':
        return UNet_vgg16(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg19':
        return UNet_vgg19(config, num_classes, weights, base_trainable)
    else:
        assert False, "Supported Backbones: 'vgg16', 'vgg19', \
        'resnet50', 'resnet101', 'resnet152', 'mobilenet'"
