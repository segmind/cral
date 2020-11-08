from abc import ABC, abstractmethod

import cv2
import numpy as np
import tensorflow as tf


def pad_resize(image, height, width, scale):
    """Summary.

    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description

    Returns:
        numpy nd.array: Description
    """

    image = image.astype(np.uint8)

    padded_image = np.zeros(
        shape=(height.astype(int), width.astype(int), 3), dtype=np.uint8)
    h, w, _ = image.shape
    padded_image[:h, :w, :] = image
    resized_image = cv2.resize(padded_image, None, fx=scale, fy=scale)

    return resized_image


@tf.function
def decode_pad_resize(image_string, pad_height, pad_width, scale):
    """Summary.

    Args:
        image_string (TYPE): Description
        pad_height (TYPE): Description
        pad_width (TYPE): Description
        scale (TYPE): Description

    Returns:
        tf.tensor: Description
    """
    image = tf.image.decode_jpeg(image_string)
    image = tf.numpy_function(
        pad_resize, [image, pad_height, pad_width, scale], Tout=tf.uint8)
    # image.set_shape([None, None, 3])
    return image


class DetectionBase(ABC):
    """docstring for DetectionBase."""

    def __init__(self,
                 train_tfrecords,
                 test_tfrecords,
                 min_side,
                 max_side,
                 num_classes,
                 bboxes_format,
                 processing_func=lambda x: x.astype(tf.keras.backend.floatx()),
                 augmentation=None,
                 batch_size=4):

        self.train_tfrecords = train_tfrecords
        self.test_tfrecords = test_tfrecords

        self.min_side = int(min_side)
        self.max_side = int(max_side)

        self.num_classes = int(num_classes)
        self.batch_size = batch_size
        self.aug = augmentation
        self.bboxes_format = bboxes_format

        # # initialize the composer
        # if augmentation != None:
        #     self.aug = Compose(
        #       augmentation,
        #       bbox_params=BboxParams(
        #         format=self.bboxes_format,
        #         min_area=0.0,
        #         min_visibility=0.0,
        #         label_fields=['category_id'])
        #       )
        # else:
        #     self.aug = None

        self.normalize_func = processing_func

    def normalize_image(self, image):
        return self.normalize_func(image).astype(tf.keras.backend.floatx())

    @tf.function
    @abstractmethod
    def yield_image_regression_classification(self, image_batch, bboxes_batch,
                                              labels_batch):
        return image_batch, bboxes_batch, labels_batch

    @tf.function
    def normalize_image_batch(self, image):
        return tf.numpy_function(
            self.normalize_image, [image], Tout=tf.keras.backend.floatx())

    @staticmethod
    def print_array(array):
        print(array.shape)
        return array

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
            tf.io.VarLenFeature(tf.int64)
        }

        parsed_example = tf.io.parse_example(
            serialized=serialized, features=features)

        max_height = tf.cast(
            tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(
            tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        smallest_side = tf.keras.backend.minimum(max_height, max_width)
        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side
        # check if the largest side is now greater than max_side, which can
        # happen when images have a large aspect ratio
        largest_side = tf.keras.backend.maximum(max_height, max_width)

        scale = tf.cond(
            largest_side * tf.cast(scale, tf.int32) > self.max_side,
            lambda: self.max_side / largest_side, lambda: scale)
        scale = tf.cast(scale, tf.keras.backend.floatx())

        image_batch = tf.map_fn(
            lambda x: decode_pad_resize(x, max_height, max_width, scale),
            parsed_example['image/encoded'],
            dtype=tf.uint8)

        xmin_batch = tf.sparse.to_dense(
            parsed_example['image/object/bbox/xmin'] * scale, default_value=-1)
        xmax_batch = tf.sparse.to_dense(
            parsed_example['image/object/bbox/xmax'] * scale, default_value=-1)
        ymin_batch = tf.sparse.to_dense(
            parsed_example['image/object/bbox/ymin'] * scale, default_value=-1)
        ymax_batch = tf.sparse.to_dense(
            parsed_example['image/object/bbox/ymax'] * scale, default_value=-1)

        label_batch = tf.sparse.to_dense(
            parsed_example['image/object/class/label'], default_value=-1)

        augmented_image_batch, regression_batch, classification_batch = self.yield_image_regression_classification(  # noqa: E501
            xmin_batch, ymin_batch, xmax_batch, ymax_batch, label_batch,
            image_batch)

        normalized_image_batch = tf.map_fn(
            lambda x: self.normalize_image_batch(x),
            augmented_image_batch,
            dtype=tf.keras.backend.floatx())

        return normalized_image_batch, {
            'regression': regression_batch,
            'classification': classification_batch
        }

    def get_train_function(self):

        dataset = tf.data.Dataset.list_files(
            self.train_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True)  # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def get_test_function(self):

        dataset = tf.data.Dataset.list_files(
            self.test_tfrecords).shuffle(buffer_size=256).repeat(-1)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False)

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True)  # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


def print_annotations_dims(image, bboxes, labels):
    tf.print(image.shape, bboxes.shape, labels.shape)
    # tf.print(bboxes)
    return image


class TfrecordParser(DetectionBase):
    """docstring for TfrecordParser."""

    def yield_image_regression_classification(self, xmin_batch, ymin_batch,
                                              xmax_batch, ymax_batch,
                                              label_batch, image_batch):

        regression_batch = list()
        classification_batch = list()

        print('yielding dummy ....')

        for index in range(self.batch_size):
            image = image_batch[index]
            xmins, ymins, xmaxs, ymaxs, labels = xmin_batch[index], ymin_batch[
                index], xmax_batch[index], ymax_batch[index], label_batch[
                    index]
            bboxes = tf.convert_to_tensor([xmins, ymins, xmaxs, ymaxs],
                                          dtype=tf.keras.backend.floatx())
            bboxes = tf.transpose(bboxes)

            regression_batch.append(bboxes)
            classification_batch.append(labels)

            image = tf.numpy_function(
                print_annotations_dims, [image, bboxes, labels], Tout=tf.uint8)

        return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(
            classification_batch)
