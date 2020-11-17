import tensorflow as tf
import numpy as np
from .utils import MaskRCNNConfig
from .parsing_utils import (_compute_backbone_shapes,
                            _generate_pyramid_anchors,
                            _build_rpn_targets,
                            minimize_mask, _resize_image, _resize_mask,
                            _extract_bboxes, _compose_image_meta)


def backbones_anchors(config):
    backbone_shapes = _compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = _generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                        config.RPN_ANCHOR_RATIOS,
                                        backbone_shapes,
                                        config.BACKBONE_STRIDES,
                                        config.RPN_ANCHOR_STRIDE)
    return anchors


class MaskRCNNGenerator(object):
    """docstring for DeepLabv3Generator"""
    def __init__(self,
                 config,
                 train_tfrecords,
                 test_tfrecords,
                 # mask_format,
                 processing_func=lambda x: x.astype(tf.keras.backend.floatx()),
                 augmentation=None,
                 batch_size=1, num_classes=1):

        assert isinstance(config, MaskRCNNConfig), 'please provide a `MaskRCNNConfig()` object'  # noqa: E501
        self.config = config

        self.train_tfrecords = train_tfrecords
        self.test_tfrecords = test_tfrecords

        # self.min_side = int(min_side)
        # self.max_side = int(max_side)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.aug = augmentation
        # self.mask_format = mask_format

        self.normalize_func = processing_func
        self.anchors = backbones_anchors(config)

    # def parse_tfrecords(filenames, height, width, batch_size=32):

    def _get_mask(self, mask, height, width, label):
        mask = np.frombuffer(mask, dtype='bool')
        mask = mask.reshape((height, width, len(label)))
#       _par_resize(mask, )
#       print(mask.shape, label, image_batch.shape)
#       _annotate_and_save_image(image_batch, mask)
        return mask

    def _load_image_gt(self, image, mask, label, fid):
        #     print(image.shape, mask.shape, label)
        original_shape = image.shape
        image, window, scale, padding, crop = _resize_image(
          image,
          min_dim=self.config.IMAGE_MIN_DIM,
          min_scale=self.config.IMAGE_MIN_SCALE,
          max_dim=self.config.IMAGE_MAX_DIM,
          mode=self.config.IMAGE_RESIZE_MODE)

        mask = _resize_mask(mask, scale, padding, crop)
        #   print('after conversion ' + str(image.shape) + ' ' +str(mask.shape))  # noqa: E501
        _idx = np.sum(mask, axis=(0, 1)) > 0
        mask = mask[:, :, _idx]
        class_ids = label[_idx]
        bbox = _extract_bboxes(mask)
        #     print(bbox)
        if self.config.USE_MINI_MASK:
            mask = minimize_mask(bbox, mask, self.config.MINI_MASK_SHAPE)
#         print(self.num_classes)
        active_class_ids = np.ones([self.num_classes], dtype=np.int32)
        image_meta = _compose_image_meta(fid, original_shape, image.shape,
                                         window, scale, active_class_ids)
        #     print(image_meta)
        #     print(type(image), type(image_meta), type(class_ids), type(bbox),
        #     print(type(mask))
        #     return True
        return image, image_meta, class_ids, bbox, mask

    def rpn_t(self, image, gt_class_ids, gt_boxes):
        rpn_match, rpn_bbox = _build_rpn_targets(self.config, image.shape,
                                                 self.anchors,
                                                 gt_class_ids, gt_boxes)

        return rpn_match, rpn_bbox

    def _mold_image(self, images):
        return images.astype(np.float32) - self.config.MEAN_PIXEL

    def ret_final(self, image_meta, rpn_match, rpn_bbox, image, gt_class_ids,
                  gt_boxes, gt_masks):
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
        batch_gt_masks = np.zeros(
          (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
           self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)  # noqa: E501
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
        b = 0
        #     for b in range(8):
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = self._mold_image(image.astype(np.float32))
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        # print(batch_images.shape, batch_image_meta.shape)
        # print(batch_rpn_match.shape)
        # print(batch_rpn_bbox.shape, batch_gt_class_ids.shape)
        # print(batch_gt_boxes.shape, batch_gt_masks.shape)
        return batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks  # noqa: E501

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
         'image/object/mask':
         tf.io.FixedLenFeature([], tf.string)
        }

        parsed_example = tf.io.parse_single_example(
            serialized=serialized, features=features)
        fid = tf.cast(tf.keras.backend.max(parsed_example['image/f_id']), tf.int32)  # noqa: E501
        max_height = tf.cast(
            tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(
            tf.keras.backend.max(parsed_example['image/width']), tf.int32)
        image_batch = tf.image.decode_jpeg(parsed_example['image/encoded'])

        # xmin = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'],
        #                          default_value=-1)
        # xmax = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'],
        #                          default_value=-1)
        # ymin = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'],
        #                          default_value=-1)
        # ymax = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'],
        #                          default_value=-1)
        label = tf.sparse.to_dense(parsed_example['image/object/class/label'],
                                   default_value=-1)

        mask = parsed_example['image/object/mask']
        mask = tf.numpy_function(self._get_mask, [mask, max_height, max_width, label], Tout=tf.bool)  # noqa: E501

        image, image_meta, gt_class_ids, gt_boxes, gt_masks = tf.numpy_function(self._load_image_gt, [image_batch, mask, label, fid], Tout=[tf.uint8, tf.float64, tf.int64, tf.int32, tf.bool])  # noqa: E501
        rpn_match, rpn_bbox = tf.numpy_function(self.rpn_t, [image, gt_class_ids, gt_boxes], Tout=[tf.int32, tf.float64])  # noqa: E501

        batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = tf.numpy_function(self.ret_final, [image_meta, rpn_match, rpn_bbox, image, gt_class_ids, gt_boxes, gt_masks], Tout=[tf.float32, tf.float64, tf.int32, tf.float64, tf.int32, tf.int32, tf.bool])  # noqa: E501
        batch_images.set_shape([None, None, None, 3])
        batch_image_meta.set_shape([None, self.num_classes + 12])
        batch_rpn_match.set_shape([None, None, 1])
        batch_rpn_bbox.set_shape([None, None, 4])
        batch_gt_class_ids.set_shape([None, None])
        batch_gt_boxes.set_shape([None, None, 4])
        batch_gt_masks.set_shape([None, None, None, None])

        inputs = (batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks)  # noqa: E501
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
