import tensorflow as tf
from cral.common import classification_networks
from tensorflow.keras.models import Model

from .utils import FpnNetConfig


def fpn_block(feats, skip):
    input_tensor = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(1, 1),
        kernel_initializer='he_uniform',
        activation='relu')(
            feats)
    skip = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(1, 1),
        kernel_initializer='he_uniform',
        activation='relu')(
            skip)
    x = tf.keras.layers.UpSampling2D((2, 2))(input_tensor)
    x = tf.keras.layers.Add()([x, skip])
    return x


def Conv3x3BnReLU(filters, name=None):

    def wrapper(input_tensor):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            name=name)(
                input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    return wrapper


def DoubleConv3x3BnReLU(filters, name=None):
    name1, name2 = None, None
    if name is not None:
        name1 = name + 'a'
        name2 = name + 'b'

    def wrapper(input_tensor):
        x = Conv3x3BnReLU(filters, name=name1)(input_tensor)
        x = Conv3x3BnReLU(filters, name=name2)(x)
        return x

    return wrapper


def FpnNet_vgg16(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg16'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('block5_conv3').output
    f2 = base_model.get_layer('block4_conv3').output
    f3 = base_model.get_layer('block3_conv3').output
    f4 = base_model.get_layer('block2_conv2').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_vgg16')
    return model, preprocessing_function


def FpnNet_vgg19(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['vgg19'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('block5_conv4').output
    f2 = base_model.get_layer('block4_conv4').output
    f3 = base_model.get_layer('block3_conv4').output
    f4 = base_model.get_layer('block2_conv2').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_vgg19')
    return model, preprocessing_function


def FpnNet_resnet50(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block6_out').output
    f2 = base_model.get_layer('conv3_block4_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_resnet50')
    return model, preprocessing_function


def FpnNet_resnet101(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet101'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block23_out').output
    f2 = base_model.get_layer('conv3_block4_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_resnet101')
    return model, preprocessing_function


def FpnNet_resnet152(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet152'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block36_out').output
    f2 = base_model.get_layer('conv3_block8_out').output
    f3 = base_model.get_layer('conv2_block3_out').output
    f4 = base_model.get_layer('conv1_relu').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_resnet152')
    return model, preprocessing_function


def FpnNet_resnet50v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['resnet50v2'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block6_1_relu').output
    f2 = base_model.get_layer('conv3_block4_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_resnet50v2')
    return model, preprocessing_function


def FpnNet_resnet101v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet101v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block23_1_relu').output
    f2 = base_model.get_layer('conv3_block4_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(
        inputs=base_model.input, outputs=o, name='FpnNet_resnet101v2')
    return model, preprocessing_function


def FpnNet_resnet152v2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'resnet152v2'](
            input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv4_block36_1_relu').output
    f2 = base_model.get_layer('conv3_block8_1_relu').output
    f3 = base_model.get_layer('conv2_block3_1_relu').output
    f4 = base_model.get_layer('conv1_conv').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(
        inputs=base_model.input, outputs=o, name='FpnNet_resnet152v2')
    return model, preprocessing_function


def FpnNet_mobilenet(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks['mobilenet'](
        input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('conv_pw_11_relu').output
    f2 = base_model.get_layer('conv_pw_5_relu').output
    f3 = base_model.get_layer('conv_pw_3_relu').output
    f4 = base_model.get_layer('conv_pw_1_relu').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(inputs=base_model.input, outputs=o, name='FpnNet_mobilenet')
    return model, preprocessing_function


def FpnNet_mobilenetv2(config, num_classes, weights, base_trainable):
    base_model, preprocessing_function = classification_networks[
        'mobilenetv2'](
            input_shape=config.input_shape, weights=weights, include_top=False)

    base_model.trainable = base_trainable
    segmentation_filters = 128
    o = base_model.output

    f1 = base_model.get_layer('block_13_expand_relu').output
    f2 = base_model.get_layer('block_6_expand_relu').output
    f3 = base_model.get_layer('block_3_expand_relu').output
    f4 = base_model.get_layer('block_1_expand_relu').output
    skips = [f1, f2, f3, f4]

    p5 = fpn_block(o, skips[0])
    p4 = fpn_block(p5, skips[1])
    p3 = fpn_block(p4, skips[2])
    p2 = fpn_block(p3, skips[3])

    s5 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage5')(p5)
    s4 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage4')(p4)
    s3 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage3')(p3)
    s2 = DoubleConv3x3BnReLU(segmentation_filters, name='segm_stage2')(p2)

    s5 = tf.keras.layers.UpSampling2D((8, 8),
                                      interpolation='bilinear',
                                      name='upsampling_stage5')(
                                          s5)
    s4 = tf.keras.layers.UpSampling2D((4, 4),
                                      interpolation='bilinear',
                                      name='upsampling_stage4')(
                                          s4)
    s3 = tf.keras.layers.UpSampling2D((2, 2),
                                      interpolation='bilinear',
                                      name='upsampling_stage3')(
                                          s3)

    o = tf.keras.layers.Concatenate(
        axis=3, name='aggregation_concat')([s2, s3, s4, s5])
    o = tf.keras.layers.SpatialDropout2D(rate=0.2, name='pyramid_dropout')(o)
    o = Conv3x3BnReLU(segmentation_filters, name='final_stage')(o)
    o = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear', name='final_upsampling')(
            o)

    o = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='head_conv')(
            o)

    o = tf.keras.layers.Activation(activation='softmax', name='softmax')(o)

    model = Model(
        inputs=base_model.input, outputs=o, name='FpnNet_mobilenetv2')
    return model, preprocessing_function


def create_FpnNet(feature_extractor,
                  config,
                  num_classes,
                  weights,
                  base_trainable=True):
    assert isinstance(config,
                      FpnNetConfig), 'please provide a `FpnNetConfig()` object'
    if feature_extractor == 'vgg16':
        return FpnNet_vgg16(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'vgg19':
        return FpnNet_vgg19(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50':
        return FpnNet_resnet50(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101':
        return FpnNet_resnet101(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152':
        return FpnNet_resnet152(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet50v2':
        return FpnNet_resnet50v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet101v2':
        return FpnNet_resnet101v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'resnet152v2':
        return FpnNet_resnet152v2(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenet':
        return FpnNet_mobilenet(config, num_classes, weights, base_trainable)
    elif feature_extractor == 'mobilenetv2':
        return FpnNet_mobilenetv2(config, num_classes, weights, base_trainable)

    else:
        assert False, "Supported Backbones: 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'mobilenetV2', 'mobilenet', 'resnet50v2', 'resnet101v2', 'resnet152v2'"  # noqa: E501
