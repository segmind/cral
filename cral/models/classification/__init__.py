from cral.models.classification.classification_utils import (
    MLPConfig, densely_connected_head)
from cral.models.classification.darknet import \
    preprocess_input as darknet_preprocess_input
from cral.models.classification.efficientnet import \
    preprocess_input as efficientnet_preprocess_input
# from segmind import log_params_decorator
from tensorflow.keras.applications.densenet import \
    preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import \
    preprocess_input as inception_resnet_v2_preprocess_input
from tensorflow.keras.applications.inception_v3 import \
    preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.mobilenet import \
    preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import \
    preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.nasnet import \
    preprocess_input as nasnet_preprocess_input
from tensorflow.keras.applications.resnet import \
    preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.resnet_v2 import \
    preprocess_input as resnet_v2_preprocess_input
from tensorflow.keras.applications.vgg16 import \
    preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import \
    preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.xception import \
    preprocess_input as xception_preprocess_input


# @log_params_decorator
def DenseNet121(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Densenet121 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """
    from tensorflow.keras.applications.densenet import DenseNet121 as cral_DenseNet121
    return cral_DenseNet121(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), densenet_preprocess_input


# @log_params_decorator
def DenseNet169(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Densenet169 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """
    from tensorflow.keras.applications.densenet import DenseNet169 as cral_DenseNet169
    return cral_DenseNet169(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), densenet_preprocess_input


# @log_params_decorator
def DenseNet201(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    """Instantiates the Densenet201 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """
    from tensorflow.keras.applications.densenet import DenseNet201 as cral_DenseNet201
    return cral_DenseNet201(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), densenet_preprocess_input


# @log_params_decorator
def InceptionResNetV2(include_top=False,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      classifier_activation='softmax'):
    """Instantiates the Inception-ResNet v2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.inception_resnet_v2.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    Deleted Parameters:
        **kwargs: For backwards compatibility only.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as cral_InceptionResNetV2
    return cral_InceptionResNetV2(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), inception_resnet_v2_preprocess_input


# @log_params_decorator
def InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax'):
    """Instantiates the Inception v3 architecture.

    Reference paper:
        - [Rethinking the Inception Architecture for Computer Vision](
                http://arxiv.org/abs/1512.00567) (CVPR 2016)

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in the `tf.keras.backend.image_data_format()`.

        Caution: Be sure to properly pre-process your inputs to the application.
        Please see `applications.inception_v3.preprocess_input` for an example.

    Arguments:
        include_top: Boolean, whether to include the fully-connected
            layer at the top, as the last layer of the network. Default to `True`.
        weights: One of `None` (random initialization),
            `imagenet` (pre-training on ImageNet),
            or the path to the weights file to be loaded. Default to `imagenet`.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model. `input_tensor` is useful for sharing
            inputs between multiple different networks. Default to None.
        input_shape: Optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
            `input_shape` will be ignored if the `input_tensor` is provided.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` (default) means that the output of the model will be
                    the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified. Default to 1000.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.inception_v3 import InceptionV3 as cral_InceptionV3
    return cral_InceptionV3(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), inception_v3_preprocess_input


# @log_params_decorator
def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=0.001,
              include_top=False,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax'):
    """Instantiates the MobileNet architecture.

    Reference paper:
        - [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
            Applications](https://arxiv.org/abs/1704.04861)

        Optionally loads weights pre-trained on ImageNet.
        Note that the data format convention used by the model is
        the one specified in the `tf.keras.backend.image_data_format()`.

        Caution: Be sure to properly pre-process your inputs to the application.
        Please see `applications.mobilenet.preprocess_input` for an example.

    Arguments:
        input_shape: Optional shape tuple, only to be specified if `include_top`
            is False (otherwise the input shape has to be `(224, 224, 3)` (with
            `channels_last` data format) or (3, 224, 224) (with `channels_first`
            data format). It should have exactly 3 inputs channels, and width and
            height should be no smaller than 32. E.g. `(200, 200, 3)` would be one
            valid value. Default to `None`.
            `input_shape` will be ignored if the `input_tensor` is provided.
        alpha: Controls the width of the network. This is known as the width
            multiplier in the MobileNet paper. - If `alpha` < 1.0, proportionally
            decreases the number of filters in each layer. - If `alpha` > 1.0,
            proportionally increases the number of filters in each layer. - If
            `alpha` = 1, default number of filters from the paper are used at each
            layer. Default to 1.0.
        depth_multiplier: Depth multiplier for depthwise convolution. This is
            called the resolution multiplier in the MobileNet paper. Default to 1.0.
        dropout: Dropout rate. Default to 0.001.
        include_top: Boolean, whether to include the fully-connected layer at the
            top of the network. Default to `True`.
        weights: One of `None` (random initialization), 'imagenet' (pre-training
            on ImageNet), or the path to the weights file to be loaded. Default to
            `imagenet`.
        input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model. `input_tensor` is useful for sharing
            inputs between multiple different networks. Default to None.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` (default) means that the output of the model will be
                    the 4D tensor output of the last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: Optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified. Defaults to 1000.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    Deleted Parameters:
        **kwargs: For backwards compatibility only.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.mobilenet import MobileNet as cral_MobileNet
    return cral_MobileNet(
        input_shape=input_shape,
        alpha=alpha,
        depth_multiplier=depth_multiplier,
        dropout=dropout,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), mobilenet_preprocess_input


# @log_params_decorator
def MobileNetV2(input_shape=None,
                alpha=1.0,
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax'):
    """Instantiates the MobileNetV2 architecture.

    Reference paper:
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks]
        (https://arxiv.org/abs/1801.04381) (CVPR 2018)

        Optionally loads weights pre-trained on ImageNet.

        Caution: Be sure to properly pre-process your inputs to the application.
        Please see `applications.mobilenet_v2.preprocess_input` for an example.

    Arguments:
        input_shape: Optional shape tuple, to be specified if you would
            like to use a model with an input image resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: Float between 0 and 1. controls the width of the network.
            This is known as the width multiplier in the MobileNetV2 paper,
            but the name is kept for consistency with `applications.MobileNetV1`
            model in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
        include_top: Boolean, whether to include the fully-connected
            layer at the top of the network. Defaults to `True`.
        weights: String, one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: String, optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                    will be the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a
                    2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: Integer, optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    Deleted Parameters:
        **kwargs: For backwards compatibility only.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as cral_MobileNetV2
    return cral_MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), mobilenet_v2_preprocess_input


# @log_params_decorator
def NASNetLarge(input_shape=None,
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000):
    """Instantiates a NASNet model in ImageNet mode.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        input_shape: Optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(331, 331, 3)` for NASNetLarge.
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
        include_top: Whether to include the fully-connected
                layer at the top of the network.
        weights: `None` (random initialization) or
                `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
                `layers.Input()`)
                to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model
                        will be the 4D tensor output of the
                        last convolutional layer.
                - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a
                        2D tensor.
                - `max` means that global max pooling will
                        be applied.
        classes: Optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function

    No Longer Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    from tensorflow.keras.applications.nasnet import NASNetLarge as cral_NASNetLarge
    return cral_NASNetLarge(
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes), nasnet_preprocess_input


# @log_params_decorator
def NASNetMobile(input_shape=None,
                 include_top=False,
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 classes=1000):
    """Instantiates a Mobile NASNet model in ImageNet mode.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        input_shape: Optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` for NASNetMobile
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(224, 224, 3)` would be one valid value.
        include_top: Whether to include the fully-connected
                layer at the top of the network.
        weights: `None` (random initialization) or
                `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
                `layers.Input()`)
                to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model
                        will be the 4D tensor output of the
                        last convolutional layer.
                - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a
                        2D tensor.
                - `max` means that global max pooling will
                        be applied.
        classes: Optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function

    No Longer Raises:
        ValueError: In case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    from tensorflow.keras.applications.nasnet import NASNetMobile as cral_NASNetMobile
    return cral_NASNetMobile(
        input_shape=input_shape,
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes), nasnet_preprocess_input


# @log_params_decorator
def ResNet50(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """

    from tensorflow.keras.applications.resnet import ResNet50 as cral_ResNet50
    return cral_ResNet50(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), resnet_preprocess_input


# @log_params_decorator
def ResNet101(include_top=False,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    """Instantiates the ResNet101 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """
    from tensorflow.keras.applications.resnet import ResNet101 as cral_ResNet101
    return cral_ResNet101(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), resnet_preprocess_input


# @log_params_decorator
def ResNet152(include_top=False,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    """Instantiates the ResNet152 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
         A Keras model instance and the respective preprocessing function
    """
    from tensorflow.keras.applications.resnet import ResNet152 as cral_ResNet152
    return cral_ResNet152(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes), resnet_preprocess_input


# @log_params_decorator
def ResNet50V2(include_top=False,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               classifier_activation='softmax'):
    """Instantiates the ResNet50V2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet_v2.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.
    """
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as cral_ResNet50V2
    return cral_ResNet50V2(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), resnet_v2_preprocess_input


# @log_params_decorator
def ResNet101V2(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax'):
    """Instantiates the ResNet101V2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet_v2.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.
    """
    from tensorflow.keras.applications.resnet_v2 import ResNet101V2 as cral_ResNet101V2
    return cral_ResNet101V2(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), resnet_v2_preprocess_input


# @log_params_decorator
def ResNet152V2(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation='softmax'):
    """Instantiates the ResNet152V2 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet_v2.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.
    """
    from tensorflow.keras.applications.resnet_v2 import ResNet152V2 as cral_ResNet152V2
    return cral_ResNet152V2(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), resnet_v2_preprocess_input


# @log_params_decorator
def VGG16(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation='softmax'):
    """Instantiates the VGG16 model.

    By default, it loads weights pre-trained on ImageNet. Check 'weights' for
    other options.

    This model can be built both with 'channels_first' data format
    (channels, height, width) or 'channels_last' data format
    (height, width, channels).

    The default input size for this model is 224x224.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.vgg16.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the 3 fully-connected
                layers at the top of the network.
        weights: one of `None` (random initialization),
                    'imagenet' (pre-training on ImageNet),
                    or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)`
                (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 input channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional block.
                - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional block, and thus
                        the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                        be applied.
        classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
                on the "top" layer. Ignored unless `include_top=False`. Set
                `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.vgg16 import VGG16 as cral_VGG16
    return cral_VGG16(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation), vgg16_preprocess_input


# @log_params_decorator
def VGG19(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation='softmax'):
    """Instantiates the VGG19 architecture.

    By default, it loads weights pre-trained on ImageNet. Check 'weights' for
    other options.

    This model can be built both with 'channels_first' data format
    (channels, height, width) or 'channels_last' data format
    (height, width, channels).

    The default input size for this model is 224x224.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.vgg19.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
                'imagenet' (pre-training on ImageNet),
                or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.vgg19 import VGG19 as cral_VGG19
    return cral_VGG19(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation), vgg19_preprocess_input


# @log_params_decorator
def Xception(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             classifier_activation='softmax'):
    """Instantiates the Xception architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Note that the default input image size for this model is 299x299.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.xception.preprocess_input` for an example.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional block.
            - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional block, and thus
                    the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                    be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True,
            and if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=False`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    No Longer Raises:
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    from tensorflow.keras.applications.xception import Xception as cral_Xception
    return cral_Xception(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation), xception_preprocess_input


#def efficientnet_preprocess_input(x,backend=tensorflow.keras.backend,**kwargs):
#    return efficientnet_preprocess_input_org(x=x, backend=backend,**kwargs)


# @log_params_decorator
def EfficientNetB0(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB0 as cral_EfficientNetB0
    return cral_EfficientNetB0(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB1(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB1 as cral_EfficientNetB1
    return cral_EfficientNetB1(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB2(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB2 as cral_EfficientNetB2
    return cral_EfficientNetB2(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB3(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB3 as cral_EfficientNetB3
    return cral_EfficientNetB3(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB4(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB4 as cral_EfficientNetB4
    return cral_EfficientNetB4(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB5(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB5 as cral_EfficientNetB5
    return cral_EfficientNetB5(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB6(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB6 as cral_EfficientNetB6
    return cral_EfficientNetB6(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def EfficientNetB7(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   classifier_activation='softmax',
                   **kwargs):
    """Instantiates the EfficientNet architecture using given scaling
    coefficients. Optionally loads weights pre-trained on ImageNet. Note that
    the data format convention used by the model is the one specified in your
    Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .efficientnet import EfficientNetB7 as cral_EfficientNetB7
    return cral_EfficientNetB7(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation
    ), efficientnet_preprocess_input


# @log_params_decorator
def Darknet53(include_top=False,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax',
              **kwargs):
    """Instantiates the Darknet53 architecture. Optionally loads weights pre-
    trained on ImageNet. Note that the data format convention used by the model
    is the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .darknet import Darknet53 as cral_Darknet53
    return cral_Darknet53(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation), darknet_preprocess_input


# @log_params_decorator
def Detnet(include_top=False,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           classes=1000,
           pooling=None,
           classifier_activation='softmax',
           **kwargs):
    """Instantiates the Detnet architecture. Optionally loads weights pre-
    trained on ImageNet. Note that the data format convention used by the model
    is the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
            include_top: whether to include the fully-connected
                    layer at the top of the network.
            weights: one of `None` (random initialization),
                        'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
            input_tensor: optional Keras tensor
                    (i.e. output of `layers.Input()`)
                    to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                    if `include_top` is False.
                    It should have exactly 3 inputs channels.
            pooling: optional pooling mode for feature extraction
                    when `include_top` is `False`.
                    - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                    - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                    - `max` means that global max pooling will
                            be applied.
            classes: optional number of classes to classify images
                    into, only to be specified if `include_top` is True, and
                    if no `weights` argument is specified.
            classifier_activation: A `str` or callable. The activation function to use
                    on the "top" layer. Ignored unless `include_top=True`. Set
                    `classifier_activation=None` to return the logits of the "top" layer.
                    Defaults to 'softmax'.
    # Returns
             A Keras model instance and the respective preprocessing function
    # Raises
            ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """
    from .detnet import Detnet as cral_detnet
    return cral_detnet(
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation), resnet_preprocess_input
