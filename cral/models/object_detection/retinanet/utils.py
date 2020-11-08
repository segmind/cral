import numpy as np
import tensorflow as tf
from tensorflow import keras

from .postprocessing import Anchors


def log_retinanet_config_params(config):
    config_data = {}
    config_data['retinanet_sizes'] = config.sizes,
    config_data['retinanet_strides'] = config.strides
    config_data['retinanet_ratios'] = config.ratios.tolist()
    config_data['retinanet_scales'] = config.scales.tolist()
    config_data['retinanet_min_side'] = config.min_side
    config_data['retinanet_max_side'] = config.max_side


class RetinanetConfig:
    """The parameteres that define how anchors are generated.

    Args
        sizes: List of sizes to use. Each size corresponds to one feature
            level.
        strides: List of strides to use. Each stride correspond to one feature
            level.
        ratios: List of ratios to use per location in a feature map.
        scales: List of scales to use per location in a feature map.
    """

    def __init__(self,
                 sizes=[32, 64, 128, 256, 512],
                 strides=[8, 16, 32, 64, 128],
                 pyramid_levels=[3, 4, 5, 6, 7],
                 ratios=[0.5, 1, 2],
                 scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                 min_side=800,
                 max_side=1333,
                 C3=None,
                 C4=None,
                 C5=None):

        assert isinstance(
            sizes, list) and len(sizes) == 5, 'expected a list of 5 elements'
        sizes = list(map(int, sizes))

        assert isinstance(
            strides,
            list) and len(strides) == 5, 'expected a list of 5 elements'
        strides = list(map(int, strides))

        assert isinstance(ratios, list), 'expected a list'
        assert isinstance(scales, list), 'expected a list'

        min_side = int(min_side)
        max_side = int(max_side)

        self.sizes = sizes
        self.strides = strides
        self.pyramid_levels = pyramid_levels
        self.ratios = np.array(ratios, keras.backend.floatx())
        self.scales = np.array(scales, keras.backend.floatx())
        self.min_side = min_side
        self.max_side = max_side

        # layers to be used for feature extraction
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.input_anno_format = 'pascal_voc'

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)

    def check_equality(self, other):
        if isinstance(other, RetinanetConfig):
            old_dict = other.__dict__
            curr_dict = self.__dict__
            for (key1, val1), (key2, val2) in zip(old_dict.items(),
                                                  curr_dict.items()):
                if not ((key1 == key2) and str(val1) == str(val2)):
                    if key1 == key2:
                        assert False, f'Value of {key1} changes from {val1} --> {val2}'  # noqa: E501
                    else:
                        assert False, f'Value of ({key1} : {val1}) changes to ({key2} : {val2})'  # noqa: E501
                    return False
            return True
        return False


class PriorProbability(keras.initializers.Initializer):
    """Apply a prior probability to the weights.

    Attributes:
        probability (TYPE): Description
    """

    def __init__(self, probability=0.01):
        """Summary.

        Args:
            probability (float, optional): Description
        """
        super(PriorProbability, self).__init__()
        self.probability = probability

    def get_config(self):
        """Summary.

        Returns:
            TYPE: Description
        """
        return {'probability': self.probability}

    def __call__(self, shape, partition_info=None, dtype=None):
        """Summary.

        Args:
            shape (TYPE): Description
            partition_info (None, optional): Description
            dtype (None, optional): Description

        Returns:
            TYPE: Description
        """
        # set bias to -log((1 - p)/p) for foreground
        result = tf.ones(
            shape, dtype=dtype) * -tf.math.log(
                (1 - self.probability) / self.probability)
        return result


def resize_images(images, size, method='bilinear', align_corners=False):
    """
    Args:
        images (TYPE): Description
        size (TYPE): Description
        method (str, optional): One of ('bilinear', 'nearest', 'bicubic',
                'area').
        align_corners (bool, optional): Description

    Returns:
        TYPE: Description
    """
    return tf.compat.v1.image.resize_images(images, size, method,
                                            align_corners)


class UpsampleLike(keras.layers.Layer):
    """Keras layer for upsampling a Tensor to be the same shape as another
    Tensor."""

    def call(self, inputs, **kwargs):
        """Summary.

        Args:
            inputs (TYPE): Description
            **kwargs: Description

        Returns:
            TYPE: Description
        """
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            source = keras.backend.transpose(source, (0, 2, 3, 1))
            output = resize_images(
                source, (target_shape[2], target_shape[3]), method='nearest')
            output = keras.backend.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return resize_images(
                source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        """Summary.

        Args:
            input_shape (TYPE): Description

        Returns:
            TYPE: Description
        """
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0],
                    input_shape[0][1]) + input_shape[1][2:4]  # noqa: E501
        else:
            return (input_shape[0][0], ) + input_shape[1][1:3] + (
                input_shape[0][-1], )


def default_classification_model(num_classes,
                                 num_anchors,
                                 pyramid_feature_size=256,
                                 prior_probability=0.01,
                                 classification_feature_size=256,
                                 name='classification_submodel'):
    """Creates the default classification submodel.

    Args
        num_classes: Number of classes to predict a score for at each feature
                level.
        num_anchors: Number of anchors to predict classification scores for at
                each feature level.
        pyramid_feature_size: The number of filters to expect from the feature
                pyramid levels.
        classification_feature_size : The number of filters to use in the
                layers in the classification submodel.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.

    Args:
        num_classes (TYPE): Description
        num_anchors (TYPE): Description
        pyramid_feature_size (int, optional): Description
        prior_probability (float, optional): Description
        classification_feature_size (int, optional): Description
        name (str, optional): Description
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs

    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=tf.random_normal_initializer(
                mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options)(
                outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=tf.random_normal_initializer(
            mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options)(
            outputs)

    outputs = keras.layers.Reshape((-1, num_classes),
                                   name='pyramid_classification_reshape')(
                                       outputs)
    outputs = keras.layers.Activation(
        'sigmoid', name='pyramid_classification_sigmoid')(
            outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values,
                             num_anchors,
                             pyramid_feature_size=256,
                             regression_feature_size=256,
                             name='regression_submodel'):
    """Creates the default regression submodel.

    Args:
        num_values (TYPE): Number of values to regress.
        num_anchors (TYPE): Number of anchors to regress for each feature
                level.
        pyramid_feature_size (int, optional): The number of filters to expect
                from the feature pyramid levels.
        regression_feature_size (int, optional): The number of filters to use
                in the layers in the regression submodel.
        name (str, optional): The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size':
        3,
        'strides':
        1,
        'padding':
        'same',
        'kernel_initializer':
        tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer':
        'zeros'
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options)(
                outputs)

    outputs = keras.layers.Conv2D(
        num_anchors * num_values, name='pyramid_regression', **options)(
            outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1),
                                       name='pyramid_regression_permute')(
                                           outputs)
    outputs = keras.layers.Reshape((-1, num_values),
                                   name='pyramid_regression_reshape')(
                                       outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """Creates the FPN layers on top of the backbone features.

    Args
        C3: Feature stage C3 from the backbone.
        C4: Feature stage C4 from the backbone.
        C5: Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature
                levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(
        feature_size,
        kernel_size=1,
        strides=1,
        padding='same',
        name='C5_reduced')(
            C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(
        feature_size, kernel_size=3, strides=1, padding='same', name='P5')(
            P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(
        feature_size,
        kernel_size=1,
        strides=1,
        padding='same',
        name='C4_reduced')(
            C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(
        feature_size, kernel_size=3, strides=1, padding='same', name='P4')(
            P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(
        feature_size,
        kernel_size=1,
        strides=1,
        padding='same',
        name='C3_reduced')(
            C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(
        feature_size, kernel_size=3, strides=1, padding='same', name='P3')(
            P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(
        feature_size, kernel_size=3, strides=2, padding='same', name='P6')(
            C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(
        feature_size, kernel_size=3, strides=2, padding='same', name='P7')(
            P7)

    return [P3, P4, P5, P6, P7]


def default_submodels(num_classes, num_anchors):
    """Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification
    submodel.

    Args:
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns:
        A list of tuple, where the first element is the name of the submodel
        and the second element is the submodel itself.
    """
    return [('regression', default_regression_model(4, num_anchors)),
            ('classification',
             default_classification_model(num_classes, num_anchors))]


def __build_model_pyramid(name, model, features):
    """Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features

    Args:
        name (TYPE): Description
        model (TYPE): Description
        features (TYPE): Description

    Returns:
        TYPE: Description
    """
    return keras.layers.Concatenate(
        axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """Applies all submodels to each FPN level.

    Args
        models : List of submodels to run on each pyramid level (by default
                only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.

    Args:
        models (TYPE): Description
        features (TYPE): Description

    Returns:
        TYPE: Description
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(features, sizes, strides, ratios, scales):
    """Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are
                    sgenerated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```

    Args:
        features (TYPE): Description
        sizes (TYPE, optional): Description
        strides (TYPE, optional): Description
        ratios (TYPE, optional): Description
        scales (TYPE, optional): Description

    Returns:
        TYPE: Description
    """
    anchors = [
        Anchors(
            size=sizes[i],
            stride=strides[i],
            ratios=ratios,
            scales=scales,
            name='anchors_{}'.format(i))(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)
