import cv2
import numpy as np
import tensorflow as tf

from .parsing_utils import (_build_rpn_targets, _compose_image_meta,
                            _compute_backbone_shapes,
                            _generate_pyramid_anchors, _resize_image)
from .utils import FasterRCNNConfig


def backbones_anchors(config):
    backbone_shapes = _compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = _generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                        config.RPN_ANCHOR_RATIOS,
                                        backbone_shapes,
                                        config.BACKBONE_STRIDES,
                                        config.RPN_ANCHOR_STRIDE)
    return anchors


def pad_resize(image, height, width, resize_width, resize_height):
    """Summary
    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description
    Returns:
        numpy nd.array: Description
    """
    padded_image = np.zeros(shape=(height.astype(int), width.astype(int), 3),
                            dtype=image.dtype)
    h, w, _ = image.shape
    padded_image[:h, :w, :] = image
    resized_image = cv2.resize(padded_image, (resize_width, resize_height)).astype(tf.keras.backend.floatx())  # noqa: E501
    return resized_image


@tf.function
def decode_pad_resize(image_string, pad_height, pad_width, resize_width, resize_height):  # noqa: E501
    """Summary
    Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      esize_width, resize_height (TYPE): Description
    Returns:
      tf.tensor: Description
    """
    image = tf.image.decode_jpeg(image_string)
    image = tf.numpy_function(pad_resize, [image, pad_height, pad_width,
                              resize_width, resize_height],
                              Tout=tf.keras.backend.floatx())
    # image.set_shape([None, None, 3])
    return image


class FasterRCNNGenerator(object):
    """docstring for FasterRCNNGenerator."""
    def __init__(self,
                 config,
                 train_tfrecords,
                 test_tfrecords,
                 processing_func=lambda x: x.astype(tf.keras.backend.floatx()),
                 augmentation=None,
                 batch_size=1, num_classes=1):

        assert isinstance(config, FasterRCNNConfig), 'please provide a `FasterRCNNConfig()` object'  # noqa: E501
        self.config = config

        self.train_tfrecords = train_tfrecords
        self.test_tfrecords = test_tfrecords

        # self.min_side = int(min_side)
        # self.max_side = int(max_side)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.aug = augmentation

        self.normalize_func = processing_func
        self.anchors = backbones_anchors(config)

    # def parse_tfrecords(filenames, height, width, batch_size=32):

    def _load_image_gt(self, image, label, fid):
        # print(image.shape, mask.shape, label)
        original_shape = image.shape
        image, window, scale, padding, crop = _resize_image(
          image,
          min_dim=self.config.IMAGE_MIN_DIM,
          min_scale=self.config.IMAGE_MIN_SCALE,
          max_dim=self.config.IMAGE_MAX_DIM,
          mode=self.config.IMAGE_RESIZE_MODE)

        active_class_ids = np.ones([self.num_classes], dtype=np.int32)
        image_meta = _compose_image_meta(fid, original_shape, image.shape,
                                         window, scale, active_class_ids)
        #     print(image_meta)
        # print(type(image), type(image_meta), type(class_ids))
        # print(type(bbox), type(mask))
        #     return True
        return image, image_meta

    def rpn_t(self, image, gt_class_ids, gt_boxes):
        rpn_match, rpn_bbox = _build_rpn_targets(self.config, image.shape,
                                                 self.anchors,
                                                 gt_class_ids, gt_boxes)
        return rpn_match, rpn_bbox

    def _mold_image(self, images):
        return images.astype(np.float32) - self.config.MEAN_PIXEL

    def ret_final(self, image_meta, rpn_match, rpn_bbox, image, gt_class_ids, gt_boxes):  # noqa: E501
        #     print(image.shape)
        batch_image_meta = np.zeros(
                    (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)  # noqa: E501
        batch_rpn_match = np.zeros(
          [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
        batch_rpn_bbox = np.zeros(
          [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)  # noqa: E501
        batch_images = np.zeros(
          (self.batch_size,) + image.shape, dtype=np.float32)
        batch_gt_class_ids = np.zeros(
          (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
        batch_gt_boxes = np.zeros(
          (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)

        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)  # noqa: E501
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]

        b = 0
        #     for b in range(8):
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = self._mold_image(image.astype(np.float32))
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes

        return batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes   # noqa: E501

    def _parse_function(self, serialized):

        features = {
            'image/height':
            tf.io.FixedLenFeature([], tf.int64),
            'image/width':
            tf.io.FixedLenFeature([], tf.int64),
            'image/encoded':
            tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin':
            tf.io.VarLenFeature(tf.keras.backend.floatx()),
            'image/object/bbox/xmax':
            tf.io.VarLenFeature(tf.keras.backend.floatx()),
            'image/object/bbox/ymin':
            tf.io.VarLenFeature(tf.keras.backend.floatx()),
            'image/object/bbox/ymax':
            tf.io.VarLenFeature(tf.keras.backend.floatx()),
            'image/f_id':
            tf.io.FixedLenFeature([], tf.int64),
            'image/object/class/label':
            tf.io.VarLenFeature(tf.int64),
        }

        parsed_example = tf.io.parse_single_example(
            serialized=serialized, features=features)
        fid = tf.cast(
            tf.keras.backend.max(parsed_example['image/f_id']), tf.int32)
        max_height = tf.cast(
            tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(
            tf.keras.backend.max(parsed_example['image/width']), tf.int32)
        image_batch = tf.image.decode_jpeg(parsed_example['image/encoded'])

        # xmin = tf.sparse.to_dense(
        #     parsed_example['image/object/bbox/xmin'], default_value=-1)
        # xmax = tf.sparse.to_dense(
        #     parsed_example['image/object/bbox/xmax'], default_value=-1)
        # ymin = tf.sparse.to_dense(
        #     parsed_example['image/object/bbox/ymin'], default_value=-1)
        # ymax = tf.sparse.to_dense(
        #     parsed_example['image/object/bbox/ymax'], default_value=-1)
        label = tf.sparse.to_dense(
            parsed_example['image/object/class/label'], default_value=-1)

        height_scale = self.config.IMAGE_SHAPE[1]/max_height
        width_scale = self.config.IMAGE_SHAPE[0]/max_width

        height_scale = tf.keras.backend.cast_to_floatx(height_scale)
        width_scale = tf.keras.backend.cast_to_floatx(width_scale)

        # **[1] pad with -1 to batch properly
        xmin_batch = tf.expand_dims(
            tf.sparse.to_dense(
                parsed_example['image/object/bbox/xmin']*width_scale,
                default_value=-1), axis=-1)
        xmax_batch = tf.expand_dims(
            tf.sparse.to_dense(
                parsed_example['image/object/bbox/xmax']*width_scale,
                default_value=-1), axis=-1)
        ymin_batch = tf.expand_dims(
            tf.sparse.to_dense(
                parsed_example['image/object/bbox/ymin']*height_scale,
                default_value=-1), axis=-1)
        ymax_batch = tf.expand_dims(
            tf.sparse.to_dense(
                parsed_example['image/object/bbox/ymax']*height_scale,
                default_value=-1), axis=-1)

        label_batch = tf.expand_dims(
            tf.sparse.to_dense(
                parsed_example['image/object/class/label'],
                default_value=-1), axis=-1)
        label_batch = tf.keras.backend.cast_to_floatx(label_batch)

        # print(label_batch.shape, xmin_batch.shape, xmax_batch.shape)
        # print(ymin_batch.shape, ymax_batch.shape)

        # annotation_batch = tf.concat([xmin_batch, ymin_batch,
        #                               xmax_batch, ymax_batch], axis=-1)
        annotation_batch = tf.concat([ymin_batch, xmin_batch,
                                      ymax_batch, xmax_batch], axis=-1)
        # print(annotation_batch.shape)
        # y = tf.numpy_function(hello, [image_batch, mask, xmin,
        #                               xmax, ymin, ymax, label],
        #                               Tout = tf.bool)
        image, image_meta = tf.numpy_function(self._load_image_gt,
                                              [image_batch, label, fid],
                                              Tout=[tf.uint8, tf.float64])
        rpn_match, rpn_bbox = tf.numpy_function(self.rpn_t,
                                                [image, label, annotation_batch],  # noqa: E501
                                                Tout=[tf.int32, tf.float64])

        batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes = tf.numpy_function(self.ret_final, [image_meta, rpn_match, rpn_bbox, image, label, annotation_batch], Tout=[tf.float32, tf.float64, tf.int32, tf.float64, tf.int32, tf.int32])  # noqa: E501
        batch_images.set_shape([None, None, None, 3])
        batch_image_meta.set_shape([None, self.num_classes + 12])
        batch_rpn_match.set_shape([None, None, 1])
        batch_rpn_bbox.set_shape([None, None, 4])
        batch_gt_class_ids.set_shape([None, None])
        # label_batch.set_shape([None, None])
        batch_gt_boxes.set_shape([None, None, 4])

        inputs = (batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes)  # noqa: E501
        outputs = ()

        return inputs, outputs

    def get_train_function(self):

        filenames = tf.io.gfile.glob(self.train_tfrecords)
        dataset = tf.data.Dataset.from_tensor_slices(
            filenames).shuffle(buffer_size=16).repeat(-1)

        dataset = tf.data.Dataset.list_files(
            self.train_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=4,
            block_length=16)

        # dataset = dataset.batch(
        #   self.batch_size,
        #   drop_remainder=True)    # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def get_test_function(self):

        filenames = tf.io.gfile.glob(self.test_tfrecords)
        dataset = tf.data.Dataset.from_tensor_slices(
            filenames).shuffle(buffer_size=16).repeat(-1)
        dataset = tf.data.Dataset.list_files(
            self.test_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=4,
            block_length=16)

        # dataset = dataset.batch(
        #   self.batch_size,
        #   drop_remainder=True)    # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
