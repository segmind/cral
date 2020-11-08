import numpy as np
import tensorflow as tf
from cral.models.semantic_segmentation.PspNet.utils import PspNetConfig


def _pad(image, height, width, channels=3):
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
        shape=(height.astype(int), width.astype(int), channels),
        dtype=np.uint8)
    h, w, _ = image.shape
    padded_image[:h, :w, :] = image
    return padded_image


@tf.function
def decode_pad_img(image_string, pad_height, pad_width):
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
        _pad, [image, pad_height, pad_width], Tout=tf.uint8)
    image = tf.cast(image, tf.keras.backend.floatx())
    return image


@tf.function
def decode_pad_msk(mask_string, pad_height, pad_width):
    """Summary.

    Args:
        mask_string (TYPE): Description
        pad_height (TYPE): Description
        pad_width (TYPE): Description
        scale (TYPE): Description

    Returns:
        tf.tensor: Description
    """
    mask = tf.image.decode_png(mask_string)
    mask = tf.numpy_function(
        _pad, [mask, pad_height, pad_width, 1], Tout=tf.uint8)
    return mask


class PspNetGenerator(object):
    """docstring for DeepLabv3Generator."""

    def __init__(
            self,
            config,
            train_tfrecords,
            test_tfrecords,
            # num_classes,
            # mask_format,
            processing_func=lambda x: x.astype(tf.keras.backend.floatx()),
            augmentation=None,
            batch_size=4):

        assert isinstance(
            config, PspNetConfig), 'please provide a `PspNetConfig()` object'
        self.config = config

        self.train_tfrecords = train_tfrecords
        self.test_tfrecords = test_tfrecords

        # self.min_side = int(min_side)
        # self.max_side = int(max_side)

        # self.num_classes = int(num_classes)
        self.batch_size = batch_size
        self.aug = augmentation
        # self.mask_format = mask_format

        self.normalize_func = processing_func

    # def parse_tfrecords(filenames, height, width, batch_size=32):

    def _parse_function(self, serialized):

        features = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'mask/height': tf.io.FixedLenFeature([], tf.int64),
            'mask/width': tf.io.FixedLenFeature([], tf.int64),
            'mask/depth': tf.io.FixedLenFeature([], tf.int64),
            'mask_raw': tf.io.FixedLenFeature([], tf.string)
        }

        parsed_example = tf.io.parse_example(
            serialized=serialized, features=features)

        max_height = tf.cast(
            tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(
            tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        image_batch = tf.map_fn(
            lambda x: decode_pad_img(x, max_height, max_width),
            parsed_example['image_raw'],
            dtype=tf.keras.backend.floatx())
        image_batch = tf.numpy_function(
            self.normalize_func, [image_batch], Tout=tf.keras.backend.floatx())
        image_batch.set_shape([None, None, None, 3])

        mask_batch = tf.map_fn(
            lambda x: decode_pad_msk(x, max_height, max_width),
            parsed_example['mask_raw'],
            dtype=tf.uint8)
        mask_batch.set_shape([None, None, None, 1])

        # image = tf.cast(tf.image.decode_jpeg(image_string), tf.uint8)
        image_batch = tf.image.resize(image_batch,
                                      (self.config.height, self.config.width))
        image_batch = tf.cast(image_batch, tf.keras.backend.floatx())

        mask_batch = tf.image.resize(
            mask_batch, (self.config.height, self.config.width),
            method='nearest')

        return image_batch, mask_batch

    def get_train_function(self):

        filenames = tf.io.gfile.glob(self.train_tfrecords)
        dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
            buffer_size=16).repeat(-1)

        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=4,
            block_length=16)

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True)  # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def get_test_function(self):

        filenames = tf.io.gfile.glob(self.train_tfrecords)
        dataset = tf.data.Dataset.from_tensor_slices(filenames).shuffle(
            buffer_size=16).repeat(-1)
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            cycle_length=4,
            block_length=16)

        dataset = dataset.batch(
            self.batch_size, drop_remainder=True)  # Batch Size

        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
