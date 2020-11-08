from tensorflow import keras


def res_block(x, filters_list, strides=1, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + x
    # Conv2D default arguments:
        strides=1
        padding='valid'
        data_format='channels_last'
        dilation_rate=1
        activation=None
        use_bias=True
    '''
    out = keras.layers.Conv2D(
        filters=filters_list[0],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_1' % (name))(
            x)
    out = keras.layers.BatchNormalization(name='%s_1_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_1_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[1],
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=True,
        name='%s_2' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_2_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_2_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_3' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_3_bn' % (name))(out)

    out = keras.layers.Add(name='%s_add' % (name))([x, out])
    out = keras.layers.ReLU(name='%s_relu' % (name))(out)
    return out


def res_block_proj(x, filters_list, strides=2, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + proj(x)
    '''
    out = keras.layers.Conv2D(
        filters=filters_list[0],
        kernel_size=1,
        strides=strides,
        use_bias=True,
        name='%s_1' % (name))(
            x)
    out = keras.layers.BatchNormalization(name='%s_1_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_1_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[1],
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=True,
        name='%s_2' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_2_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_2_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_3' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_3_bn' % (name))(out)

    x = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=strides,
        use_bias=True,
        name='%s_proj' % (name))(
            x)
    x = keras.layers.BatchNormalization(name='%s_proj_bn' % (name))(x)

    out = keras.layers.Add(name='%s_add' % (name))([x, out])
    out = keras.layers.ReLU(name='%s_relu' % (name))(out)
    return out


def dilated_res_block(x, filters_list, strides=1, use_bias=True, name=None):
    '''
    y = f3(f2(f1(x))) + x
    '''
    out = keras.layers.Conv2D(
        filters=filters_list[0],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_1' % (name))(
            x)
    out = keras.layers.BatchNormalization(name='%s_1_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_1_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[1],
        kernel_size=3,
        strides=1,
        padding='same',
        dilation_rate=2,
        use_bias=True,
        name='%s_2' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_2_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_2_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_3' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_3_bn' % (name))(out)

    out = keras.layers.Add(name='%s_add' % (name))([x, out])
    out = keras.layers.ReLU(name='%s_relu' % (name))(out)
    return out


def dilated_res_block_proj(x,
                           filters_list,
                           strides=1,
                           use_bias=True,
                           name=None):
    '''
    y = f3(f2(f1(x))) + proj(x)
    '''
    out = keras.layers.Conv2D(
        filters=filters_list[0],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_1' % (name))(
            x)
    out = keras.layers.BatchNormalization(name='%s_1_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_1_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[1],
        kernel_size=3,
        strides=1,
        padding='same',
        dilation_rate=2,
        use_bias=True,
        name='%s_2' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_2_bn' % (name))(out)
    out = keras.layers.ReLU(name='%s_2_relu' % (name))(out)

    out = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_3' % (name))(
            out)
    out = keras.layers.BatchNormalization(name='%s_3_bn' % (name))(out)

    x = keras.layers.Conv2D(
        filters=filters_list[2],
        kernel_size=1,
        strides=1,
        use_bias=True,
        name='%s_proj' % (name))(
            x)
    x = keras.layers.BatchNormalization(name='%s_proj_bn' % (name))(x)

    out = keras.layers.Add(name='%s_add' % (name))([x, out])
    out = keras.layers.ReLU(name='%s_relu' % (name))(out)
    return out


def resnet_body(x, filters_list, num_blocks, strides=2, name=None):
    out = res_block_proj(
        x=x, filters_list=filters_list, strides=strides, name='%s_1' % (name))
    for i in range(1, num_blocks):
        out = res_block(
            x=out,
            filters_list=filters_list,
            name='%s_%s' % (name, str(i + 1)))
    return out


def detnet_body(x, filters_list, num_blocks, strides=1, name=None):
    out = dilated_res_block_proj(
        x=x, filters_list=filters_list, name='%s_1' % (name))
    for i in range(1, num_blocks):
        out = dilated_res_block(
            x=out,
            filters_list=filters_list,
            name='%s_%s' % (name, str(i + 1)))
    return out


def detnet_59(inputs, filters_list, blocks_list, num_classes, include_top):
    # stage 1
    #     inputs = keras.layers.Input(shape=(None, None, 3))
    inputs_pad = keras.layers.ZeroPadding2D(
        padding=3, name='inputs_pad')(
            inputs)
    conv1 = keras.layers.Conv2D(
        filters=filters_list[0][0],
        kernel_size=7,
        strides=2,
        use_bias=True,
        name='conv1')(
            inputs_pad)
    conv1 = keras.layers.BatchNormalization(name='conv1_bn')(conv1)
    conv1 = keras.layers.ReLU(name='conv1_relu')(conv1)

    # stage 2
    conv1_pad = keras.layers.ZeroPadding2D(padding=1, name='conv1_pad')(conv1)
    conv1_pool = keras.layers.MaxPooling2D(
        pool_size=3, strides=2, name='conv1_maxpool')(
            conv1_pad)
    conv2_x = resnet_body(
        x=conv1_pool,
        filters_list=filters_list[1],
        num_blocks=blocks_list[1],
        strides=1,
        name='res2')

    # stage 3
    conv3_x = resnet_body(
        x=conv2_x,
        filters_list=filters_list[2],
        num_blocks=blocks_list[2],
        strides=2,
        name='res3')

    # stage 4
    conv4_x = resnet_body(
        x=conv3_x,
        filters_list=filters_list[3],
        num_blocks=blocks_list[3],
        strides=2,
        name='res4')

    # stage 5
    conv5_x = detnet_body(
        x=conv4_x,
        filters_list=filters_list[4],
        num_blocks=blocks_list[4],
        strides=1,
        name='dires5')

    # stage 6
    conv6_x = detnet_body(
        x=conv5_x,
        filters_list=filters_list[5],
        num_blocks=blocks_list[5],
        strides=1,
        name='dires6')

    model = keras.Model(inputs=inputs, outputs=conv6_x)
    return model


def Detnet(input_shape=None,
           input_tensor=None,
           include_top=False,
           weights='imagenet',
           classes=1000,
           classifier_activation='softmax',
           pooling=None,
           **kwargs):

    filters_list = [[64], [64, 64, 256], [128, 128, 512], [256, 256, 1024],
                    [256, 256, 256], [256, 256, 256]]
    blocks_list = [1, 3, 4, 6, 3, 3]
    if input_shape is None and input_tensor is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    else:
        if input_shape is not None:
            assert isinstance(input_shape, tuple) and len(
                input_shape) == 3, 'input should be a tuple of 3 dimensions'
            inputs = keras.layers.Input(shape=input_shape)
        else:
            inputs = input_tensor
    model = detnet_59(inputs, filters_list, blocks_list, classes, include_top)
    if weights == 'imagenet':
        resnet = keras.applications.resnet.ResNet50(
            include_top=False, weights='imagenet')
        for i in range(143):
            model.layers[i].set_weights(resnet.layers[i].get_weights())
            model.layers[i].trainable = True
    elif weights is not None:
        model.load_weights(weights)
    return model
