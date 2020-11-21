import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from .utils import FasterRCNNConfig
from cral.common import classification_networks
from .obj_det_utils import (compute_backbone_shapes,
                            build_rpn_model,
                            ProposalLayer,
                            parse_image_meta_graph, norm_boxes_graph,
                            DetectionTargetLayer, fpn_classifier_graph,
                            rpn_class_loss_graph, rpn_bbox_loss_graph,
                            rcnn_class_loss_graph, rcnn_bbox_loss_graph,
                            DetectionLayer)
from .frcnn_utils import generate_pyramid_anchors, norm_boxes


def _get_anchors(config, image_shape):

    """Returns anchor pyramid for the given image size."""
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # Cache anchors and reuse if image shape is the same
    # if not hasattr(self, "_anchor_cache"):
    anchor_cache = {}
    a = generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
    # anchors = a
    # Normalize coordinates
    anchor_cache[tuple(image_shape)] = norm_boxes(a, image_shape[:2])

    return anchor_cache[tuple(image_shape)]


def FasterRCNN_resnet50(mode, weights, num_classes, config):
    """Build Faster R-CNN architecture.
    input_shape: The shape of the input image.
    mode: Either "training" or "inference". The inputs and
          outputs of the model differ accordingly.
    """
    assert mode in ['training', 'inference']

    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # Inputs
    input_image = KL.Input(
        shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = KL.Input(shape=[num_classes + 12],
                                name="input_image_meta")
    if mode == "training":
        # RPN GT
        input_rpn_match = KL.Input(
            shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(
            shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input(
            shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input(
            shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_gt_boxes)

    elif mode == "inference":
        # Anchors in normalized coordinates
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

    # Build the shared convolutional layers.
    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    # if callable(config.BACKBONE):
    #    _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
    #                                         train_bn=config.TRAIN_BN)
    # else:
    # _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet50",
    #                                 stage5=True,
    #                                 train_bn=config.TRAIN_BN)
    base_model, preprocessing_function = classification_networks['resnet50'](
        include_top=False,
        weights=weights,
        input_tensor=input_image)
    _ = base_model.get_layer('pool1_pool').output
    C2 = base_model.get_layer('conv2_block3_out').output
    C3 = base_model.get_layer('conv3_block4_out').output
    C4 = base_model.get_layer('conv4_block6_out').output
    C5 = base_model.get_layer('conv5_block3_out').output
    # Top-down Layers
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p2")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p3")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p4")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    rcnn_feature_maps = [P2, P3, P4, P5]

    # Anchors
    if mode == "training":
        anchors = _get_anchors(config, config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,)
                                  + anchors.shape)
        # A hack to get around Keras's bad support for constants
        # This class returns a constant layer

        class ConstLayer(tf.keras.layers.Layer):

            def __init__(self, x, name=None):
                super(ConstLayer, self).__init__(name=name)
                self.x = tf.Variable(x)

            def call(self, input):
                return self.x

        anchors = ConstLayer(anchors, name="anchors")(input_image)
    else:
        anchors = input_anchors

    # RPN Model
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS),
                          config.TOP_DOWN_PYRAMID_SIZE)
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Generate proposals
    # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    # and zero padded.
    proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    if mode == "training":

        active_class_ids = KL.Lambda(
            lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

        if not config.USE_RPN_ROIS:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                  name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_rois)
        else:
            target_rois = rpn_rois

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs and gt_boxes are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox =\
            DetectionTargetLayer(config, name="proposal_targets")([
                                 target_rois, input_gt_class_ids, gt_boxes])

        # Network Heads
        rcnn_class_logits, rcnn_class, rcnn_bbox =\
            fpn_classifier_graph(rois, rcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, num_classes,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)  # noqa: E501

        output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                   name="rpn_class_loss")(
                                   [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x),
                                  name="rpn_bbox_loss")(
                                  [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = KL.Lambda(lambda x: rcnn_class_loss_graph(*x),
                               name="rcnn_class_loss")(
                               [target_class_ids, rcnn_class_logits,
                                active_class_ids])
        bbox_loss = KL.Lambda(lambda x: rcnn_bbox_loss_graph(*x),
                              name="rcnn_bbox_loss")(
                              [target_bbox, target_class_ids, rcnn_bbox])

        # Model
        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                  input_gt_boxes]
        if not config.USE_RPN_ROIS:
            inputs.append(input_rois)
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   rcnn_class_logits, rcnn_class, rcnn_bbox,
                   rpn_rois, output_rois,
                   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
        model = KM.Model(inputs, outputs, name='Faster_rcnn')
    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        rcnn_class_logits, rcnn_class, rcnn_bbox =\
            fpn_classifier_graph(rpn_rois, rcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, num_classes,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)  # noqa: E501

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
        # in
        # normalized coordinates
        detections = DetectionLayer(config, name="Fasterrcnn_detection")(
            [rpn_rois, rcnn_class, rcnn_bbox, input_image_meta])

        model = KM.Model([input_image, input_image_meta, input_anchors],
                         [detections, rcnn_class, rcnn_bbox,
                         rpn_rois, rpn_class, rpn_bbox],
                         name='Faster_rcnn')

    return model


def FasterRCNN_resnet101(mode, weights, num_classes, config):
    """Build Faster R-CNN architecture.
    input_shape: The shape of the input image.
    mode: Either "training" or "inference". The inputs and
          outputs of the model differ accordingly.
    """
    assert mode in ['training', 'inference']

    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # Inputs
    input_image = KL.Input(
        shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = KL.Input(shape=[num_classes + 12],
                                name="input_image_meta")
    if mode == "training":
        # RPN GT
        input_rpn_match = KL.Input(
            shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(
            shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input(
            shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input(
            shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_gt_boxes)

    elif mode == "inference":
        # Anchors in normalized coordinates
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

    # Build the shared convolutional layers.
    # Bottom-up Layers
    # Returns a list of the last layers of each stage, 5 in total.
    # Don't create the thead (stage 5), so we pick the 4th item in the list.
    # if callable(config.BACKBONE):
        # _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
    #                                         train_bn=config.TRAIN_BN)
    # else:
    # _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101",
    #                                 stage5=True,
    #                                 train_bn=config.TRAIN_BN)
    base_model, preprocessing_function = classification_networks['resnet101'](
        include_top=False,
        weights=weights,
        input_tensor=input_image)
    _ = base_model.get_layer('pool1_pool').output
    C2 = base_model.get_layer('conv2_block3_out').output
    C3 = base_model.get_layer('conv3_block4_out').output
    C4 = base_model.get_layer('conv4_block23_out').output
    C5 = base_model.get_layer('conv5_block3_out').output
    # Top-down Layers
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p2")(P2)
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p3")(P3)
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p4")(P4)
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME",
                   name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    rcnn_feature_maps = [P2, P3, P4, P5]

    # Anchors
    if mode == "training":
        anchors = _get_anchors(config, config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,)
                                  + anchors.shape)
        # A hack to get around Keras's bad support for constants
        # This class returns a constant layer

        class ConstLayer(tf.keras.layers.Layer):

            def __init__(self, x, name=None):
                super(ConstLayer, self).__init__(name=name)
                self.x = tf.Variable(x)

            def call(self, input):
                return self.x

        anchors = ConstLayer(anchors, name="anchors")(input_image)
    else:
        anchors = input_anchors

    # RPN Model
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS),
                          config.TOP_DOWN_PYRAMID_SIZE)
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Generate proposals
    # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    # and zero padded.
    proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    if mode == "training":

        active_class_ids = KL.Lambda(
            lambda x: parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

        if not config.USE_RPN_ROIS:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                  name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_rois)
        else:
            target_rois = rpn_rois

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs and gt_boxes are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox =\
            DetectionTargetLayer(config, name="proposal_targets")([
                                 target_rois, input_gt_class_ids, gt_boxes])

        # Network Heads
        rcnn_class_logits, rcnn_class, rcnn_bbox =\
            fpn_classifier_graph(rois, rcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, num_classes,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)  # noqa: E501

        output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

        # Losses
        rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x),
                                   name="rpn_class_loss")(
                                   [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x),
                                  name="rpn_bbox_loss")(
                                  [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = KL.Lambda(lambda x: rcnn_class_loss_graph(*x),
                               name="rcnn_class_loss")(
                              [target_class_ids, rcnn_class_logits,
                               active_class_ids])
        bbox_loss = KL.Lambda(lambda x: rcnn_bbox_loss_graph(*x),
                              name="rcnn_bbox_loss")(
                              [target_bbox, target_class_ids, rcnn_bbox])

        # Model
        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                  input_gt_boxes]
        if not config.USE_RPN_ROIS:
            inputs.append(input_rois)
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   rcnn_class_logits, rcnn_class, rcnn_bbox,
                   rpn_rois, output_rois,
                   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
        model = KM.Model(inputs, outputs, name='Faster_rcnn')
    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        rcnn_class_logits, rcnn_class, rcnn_bbox =\
            fpn_classifier_graph(rpn_rois, rcnn_feature_maps, input_image_meta,
                                 config.POOL_SIZE, num_classes,
                                 train_bn=config.TRAIN_BN,
                                 fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)  # noqa: E501

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)]
        # in
        # normalized coordinates
        detections = DetectionLayer(config, name="Fasterrcnn_detection")(
            [rpn_rois, rcnn_class, rcnn_bbox, input_image_meta])

        model = KM.Model([input_image, input_image_meta, input_anchors],
                         [detections, rcnn_class, rcnn_bbox,
                         rpn_rois, rpn_class, rpn_bbox],
                         name='Faster_rcnn')

    return model


def create_FasterRCNN(feature_extractor, config, num_classes, weights,
                      base_trainable, mode='training'):
    assert isinstance(config, FasterRCNNConfig), 'please provide a `FasterRCNNConfig()` object'  # noqa: E501
    if feature_extractor == 'resnet50':
        return FasterRCNN_resnet50(mode, weights, num_classes, config)
    elif feature_extractor == 'resnet101':
        return FasterRCNN_resnet101(mode, weights, num_classes, config)
    else:
        assert False, "Supported Backbones: 'resnet50', 'resnet101'"
