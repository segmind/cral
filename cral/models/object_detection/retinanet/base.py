from cral.common import classification_networks
# from segmind import log_params_decorator
from tensorflow import keras

from .postprocessing import ClipBoxes, FilterDetections, RegressBoxes
from .utils import (RetinanetConfig, __build_anchors, __build_pyramid,
                    __create_pyramid_features, default_submodels)

_ALLOWED_BACKBONES = ('resnet50', 'resnet50v2', 'resnet101', 'resnet101v2',
                      'resnet152', 'resnet152v2', 'densenet121', 'densenet169',
                      'densenet201', 'mobilenet', 'mobilenetv2', 'vgg16',
                      'vgg19', 'efficientnetb0', 'efficientnetb1',
                      'efficientnetb2', 'efficientnetb3', 'efficientnetb4',
                      'efficientnetb5', 'efficientnetb6', 'xception', 'detnet')

_EFFICIENTNET_IMAGE_SIZES = [512, 640, 768, 896, 1024, 1280, 1408]

_DEFAULT_BACKBONE_FEATURE_EXTRACTOR_LAYER_NAMES = {
    'resnet50': {
        'C3': 'conv3_block4_out',
        'C4': 'conv4_block6_out',
        'C5': 'conv5_block3_out',
    },
    'resnet50v2': {
        'C3': 'conv3_block4_out',
        'C4': 'conv4_block6_out',
        'C5': 'conv5_block3_out',
    },
    'resnet101': {
        'C3': 'conv3_block4_out',
        'C4': 'conv4_block23_out',
        'C5': 'conv5_block3_out',
    },
    'resnet101v2': {
        'C3': 'conv3_block4_out',
        'C4': 'conv4_block23_out',
        'C5': 'conv5_block3_out',
    },
    'resnet152': {
        'C3': 'conv3_block8_out',
        'C4': 'conv4_block36_out',
        'C5': 'conv5_block3_out',
    },
    'resnet152v2': {
        'C3': 'conv3_block8_out',
        'C4': 'conv4_block36_out',
        'C5': 'conv5_block3_out',
    },
    'densenet121': {
        'C3': 'conv3_block12_concat',
        'C4': 'conv4_block24_concat',
        'C5': 'conv5_block16_concat',
    },
    'densenet169': {
        'C3': 'conv3_block12_concat',
        'C4': 'conv4_block32_concat',
        'C5': 'conv5_block32_concat',
    },
    'densenet201': {
        'C3': 'conv3_block12_concat',
        'C4': 'conv4_block48_concat',
        'C5': 'conv5_block32_concat',
    },
    'mobilenet': {
        'C3': 'conv_pw_5_relu',
        'C4': 'conv_pw_11_relu',
        'C5': 'conv_pw_13_relu',
    },
    'mobilenetv2': {  # guessed
        'C3':   'block_4_add',
        'C4':   'block_9_add',
        'C5':   'block_15_add',
    },
    'vgg16': {
        'C3': 'block3_pool',
        'C4': 'block4_pool',
        'C5': 'block5_pool',
    },
    'vgg19': {
        'C3': 'block3_pool',
        'C4': 'block4_pool',
        'C5': 'block5_pool',
    },
    'efficientnetb0': {
        'C3': 'block3b_add',
        'C4': 'block5c_add',
        'C5': 'block7a_project_bn'
    },
    'efficientnetb1': {
        'C3': 'block3c_add',
        'C4': 'block5d_add',
        'C5': 'block7b_add',
    },
    'efficientnetb2': {
        'C3': 'block3c_add',
        'C4': 'block5d_add',
        'C5': 'block7b_add',
    },
    'efficientnetb3': {
        'C3': 'block3c_add',
        'C4': 'block5e_add',
        'C5': 'block7b_add',
    },
    'efficientnetb4': {
        'C3': 'block3d_add',
        'C4': 'block5f_add',
        'C5': 'block7b_add',
    },
    'efficientnetb5': {
        'C3': 'block3e_add',
        'C4': 'block5g_add',
        'C5': 'block7c_add',
    },
    'efficientnetb6': {
        'C3': 'block3f_add',
        'C4': 'block5h_add',
        'C5': 'block7c_add',
    },
    'xception': {  # guessed
        'C3': 'add_3',
        'C4': 'add_7',
        'C5': 'add_11',
    },
    'detnet': {
        'C3': 'res4_6_relu',
        'C4': 'dires5_3_relu',
        'C5': 'dires6_3_relu',
    }
}


def retinanet(inputs,
              C3,
              C4,
              C5,
              num_classes,
              num_anchors,
              create_pyramid_features=__create_pyramid_features,
              submodels=None,
              name='retinanet'):
    """Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the
    unfortunate exception of anchors as output).

    Args
        inputs: keras.layers.Input (or list of) for the input to the model.
        num_classes: Number of classes to classify.
        num_anchors: Number of base anchors.
        create_pyramid_features: Functor for creating pyramid features given
                    the features C3, C4, C5 from the backbone.
        submodels: Submodels to run on each feature map (default is regression
                and classification submodels).
        name: Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs
        generated anchors and the result from each submodel on every pyramid
        level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```

    Args:
        inputs (TYPE): Description
        backbone_layers (TYPE): Description
        num_classes (TYPE): Description
        num_anchors (TYPE): Description
        create_pyramid_features (TYPE, optional): Description
        submodels (None, optional): Description
        name (str, optional): Description
    """

    if submodels is None:
        # Regression head(keras.Model with inp (None,None,256)) and
        # classification head(keras.Model with inp (None,None,256))
        submodels = default_submodels(num_classes, num_anchors)

    # C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    # i.e [P3, P4, P5, P6, P7] where each layer has 256 channels
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


# @log_params_decorator
def get_retinanet(feature_extractor,
                  num_classes,
                  num_anchors_per_location,
                  weights='imagenet',
                  base_trainable=True,
                  C3_name=None,
                  C4_name=None,
                  C5_name=None):
    """Summary.

    Args:
        num_classes (TYPE): Description
        num_anchors_per_location (TYPE, optional): Description

    Returns:
        TYPE: Description
    """

    feature_extractor = feature_extractor.lower()

    assert feature_extractor in _ALLOWED_BACKBONES, '`{}` feature extractor\
    is not yet supported, choose from below :: {}'.format(
        feature_extractor, '\n'.join(_ALLOWED_BACKBONES))

    if feature_extractor in ('efficientnetb0', 'efficientnetb1',
                             'efficientnetb2', 'efficientnetb3',
                             'efficientnetb4', 'efficientnetb5',
                             'efficientnetb6'):
        phi = int(feature_extractor.replace('efficientnetb', ''))
        input_size = _EFFICIENTNET_IMAGE_SIZES[phi]
        inputs = keras.layers.Input(shape=(input_size, input_size, 3))

    else:
        inputs = keras.layers.Input(shape=(None, None, 3))

    feature_extractor_model, preprocessing_function = classification_networks[
        feature_extractor](
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling=None)

    # freeze/train the backbone
    feature_extractor_model.trainable = base_trainable

    if C3_name is None:
        C3_name = _DEFAULT_BACKBONE_FEATURE_EXTRACTOR_LAYER_NAMES[
            feature_extractor]['C3']

    if C4_name is None:
        C4_name = _DEFAULT_BACKBONE_FEATURE_EXTRACTOR_LAYER_NAMES[
            feature_extractor]['C4']

    if C5_name is None:
        C5_name = _DEFAULT_BACKBONE_FEATURE_EXTRACTOR_LAYER_NAMES[
            feature_extractor]['C5']

    # print(feature_extractor_model)

    # C2 = resnet.get_layer('conv2_block3_out').output
    C3 = feature_extractor_model.get_layer(C3_name).output
    C4 = feature_extractor_model.get_layer(C4_name).output
    C5 = feature_extractor_model.get_layer(C5_name).output

    return retinanet(
        inputs,
        C3,
        C4,
        C5,
        num_classes=num_classes,
        num_anchors=num_anchors_per_location,
        name='{}_retinanet'.format(feature_extractor)), preprocessing_function


def get_retinanet_fromconfig(feature_extractor,
                             num_classes,
                             config,
                             weights='imagenet',
                             base_trainable=True):
    return get_retinanet(
        feature_extractor,
        num_classes=num_classes,
        num_anchors_per_location=config.num_anchors(),
        weights=weights,
        base_trainable=base_trainable,
        C3_name=config.C3,
        C4_name=config.C4,
        C5_name=config.C5)


def get_prediction_model(model, config):

    if sorted(model.output_names) == ['classification', 'regression']:

        assert isinstance(config, RetinanetConfig), 'Expected an instance of \
        cral.models.object_detection.RetinanetConfig'

        return retinanet_bbox(
            model,
            config.sizes,
            config.strides,
            config.ratios,
            config.scales,
            name=model.name)

    return model


def retinanet_bbox(
    model,
    sizes,
    strides,
    ratios,
    scales,
    name,
    class_specific_filter=True,
    nms_threshold=0.5,
    max_detections=300,
    # parallel_iterations   = 32,
):
    """Construct a RetinaNet model on top of a backbone and adds convenience
    functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to
    compute boxes within the graph. These layers include applying the
    regression values to the anchors and performing NMS.

    Args
        model: RetinaNet model to append bbox layers to. If None, it will
            create a RetinaNet model using **kwargs.
        nms: Whether to use non-maximum suppression for the filtering step.
        class_specific_filter: Whether to use class specific filtering or
            filter for the best scoring class only.
        name: Name of the model.
        anchor_params: Struct containing anchor parameters. If None, default
            values are used.
        nms_threshold: Threshold for the IoU value to determine when a box
            should be suppressed.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        parallel_iterations: Number of batch items to process in parallel.
        **kwargs: Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the
        detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values

    # compute the anchors
    features = [
        model.get_layer(p_name).output
        for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']
    ]

    anchors = __build_anchors(
        features, sizes=sizes, strides=strides, ratios=ratios, scales=scales)

    # we expect the anchors, regression and classification values as first
    # output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=True,
        class_specific_filter=class_specific_filter,
        name='filtered_detections',
        nms_threshold=nms_threshold,
        score_threshold=0.05,
        max_detections=max_detections,
        parallel_iterations=32)([boxes, classification])

    # construct the model
    return keras.models.Model(
        inputs=model.inputs, outputs=detections, name=name)
