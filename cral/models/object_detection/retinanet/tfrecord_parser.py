import numpy as np
import tensorflow as tf
from cral.data_feeder.object_detection_parser import DetectionBase

from .preprocessing import anchor_targets_bbox, anchors_for_shape
from .utils import RetinanetConfig


class RetinanetGenerator(DetectionBase):
    """docstring for RetinanetGenerator."""

    def __init__(self, config, *args, **kwargs):
        super(RetinanetGenerator, self).__init__(*args, **kwargs)
        assert isinstance(
            config,
            RetinanetConfig), 'please provide a `RetinanetConfig()` object'
        self.config = config

    def process_bboxes_labels(self, image_array, bboxes, labels):

        # delete bboxes containing [-1,-1,-1,-1]
        bboxes = bboxes[~np.all(bboxes < 0, axis=-1)]

        # delete labels containing[-1]
        labels = labels[labels > -1]

        # augment here
        if self.aug is not None:
            image_array, bboxes, labels = self.aug.apply(
                image_array, bboxes, labels)
            image_array = image_array.astype(np.uint8)

        # generate raw anchors
        raw_anchors = anchors_for_shape(
            image_shape=image_array.shape,
            sizes=self.config.sizes,
            ratios=self.config.ratios,
            scales=self.config.scales,
            strides=self.config.strides,
            pyramid_levels=self.config.pyramid_levels,
            shapes_callback=None)

        # generate anchorboxes and class labels
        gt_regression, gt_classification = anchor_targets_bbox(
            anchors=raw_anchors,
            image=image_array,
            bboxes=bboxes,
            gt_labels=labels,
            num_classes=self.num_classes,
            negative_overlap=0.4,
            positive_overlap=0.5)

        return image_array, gt_regression, gt_classification

    def yield_image_regression_classification(self, xmin_batch, ymin_batch,
                                              xmax_batch, ymax_batch,
                                              label_batch, image_batch):

        regression_batch = list()
        classification_batch = list()
        image_batch_aug = list()

        for index in range(self.batch_size):
            xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[
                index], xmax_batch[index], ymax_batch[index], label_batch[
                    index]
            image_array = image_batch[index]
            bboxes = tf.convert_to_tensor([xmins, ymins, xmaxs, ymaxs],
                                          dtype=tf.keras.backend.floatx())
            bboxes = tf.transpose(bboxes)
            augmented_image, gt_regression, gt_classification = tf.numpy_function(  # noqa: E501
                self.process_bboxes_labels, [image_array, bboxes, labels],
                Tout=[
                    tf.uint8,
                    tf.keras.backend.floatx(),
                    tf.keras.backend.floatx()
                ])

            regression_batch.append(gt_regression)
            classification_batch.append(gt_classification)
            image_batch_aug.append(augmented_image)

        return tf.convert_to_tensor(image_batch_aug), tf.convert_to_tensor(
            regression_batch), tf.convert_to_tensor(classification_batch)
