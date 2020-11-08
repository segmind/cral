import json
import math
import os
import sys
import tempfile

import cv2
import pandas as pd
import tensorflow as tf
from PIL import Image

from .utils import _bytes_feature, _float_feature, _int64_feature

# from cral.augmentations.engine import Classification as Classification_augmentor  # noqa: E501

_NUM_SHARDS = 4
_PNG_CONVERT_PATH = os.path.join(tempfile.gettempdir(), 'temp.jpg')


def get_label_dict(label_list):
    label_list.sort()
    label_dict = dict()
    for index, label in enumerate(label_list):
        label_dict[label] = index

    return label_dict


def statistics_example(data):
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': _float_feature(data['height']),
                'image/width': _float_feature(data['width']),
            }))

    return tf_example


def image_example(image_string, label, image_shape):
    feature = {
        'image_raw': _bytes_feature(image_string),
        'image/height': _int64_feature(image_shape[0]),
        'image/width': _int64_feature(image_shape[1]),
        'image_label': _int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecords(meta_json, dataset_csv, out_path):

    assert os.path.isdir(out_path)

    tfrecord_path = os.path.join(out_path, 'tfrecords')
    os.makedirs(tfrecord_path, exist_ok=True)

    with open(meta_json, 'r') as f:
        meta_info = json.loads(f.read())

    meta_info['num_training_images'] = None
    meta_info['num_test_images'] = None
    meta_info['tfrecord_path'] = out_path

    dataset_dir = meta_info['train_images_dir']

    class_list = meta_info['class_names']
    class_dict = get_label_dict(class_list)

    dataset_df = pd.read_csv(dataset_csv)

    print('processing training set ...')

    train_set = dataset_df[dataset_df['train_only'] == True]  # noqa: E712

    training_images_num = len(train_set)
    meta_info['num_training_images'] = training_images_num
    training_images_per_shard = int(
        math.ceil(training_images_num / _NUM_SHARDS))

    shard_id = 0
    num_images_seen = 0

    output_filename = os.path.join(
        out_path,
        '%s-%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS))

    tfrecord_writer = tf.io.TFRecordWriter(output_filename)

    label_file_pointer = dict()

    for index, row in train_set.iterrows():

        num_images_seen += 1

        if num_images_seen % training_images_per_shard == 0:
            tfrecord_writer.close()

            shard_id += 1

            output_filename = os.path.join(
                out_path,
                '%s-%05d-of-%05d.tfrecord' % ('train', shard_id, _NUM_SHARDS))

            tfrecord_writer = tf.io.TFRecordWriter(output_filename)

        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                         (num_images_seen, training_images_num, shard_id))
        sys.stdout.flush()

        label_str = row['annotation_name']
        image_file = os.path.join(dataset_dir, label_str, row['image_name'])

        if label_str not in label_file_pointer:
            os.makedirs(os.path.join(out_path, 'statistics'), exist_ok=True)
            label_file_pointer[label_str] = tf.io.TFRecordWriter(
                os.path.join(out_path, 'statistics', label_str) + '.tfrecord')

        assert image_file.endswith(
            ('.jpg', '.jpeg', '.png')
        ), 'required `.jpg or .jpeg or .png ` image got instead {}'.format(
            image_file)

        if image_file.endswith('.png'):
            im = Image.open(image_file)
            im.save(_PNG_CONVERT_PATH)
            image_file = _PNG_CONVERT_PATH

        with open(image_file, 'rb') as imgfile:
            image_string = imgfile.read()

        image_array = cv2.imread(image_file)
        image_shape = image_array.shape

        tf_example = image_example(
            image_string=image_string,
            label=class_dict[label_str],
            image_shape=image_shape)

        stats_feature = {
            'height': image_shape[0],
            'width': image_shape[1],
        }

        stats_example = statistics_example(stats_feature)
        label_file_pointer[label_str].write(stats_example.SerializeToString())

        tfrecord_writer.write(tf_example.SerializeToString())

    tfrecord_writer.close()

    for label_stats_pointer_file in label_file_pointer.values():
        label_stats_pointer_file.close()

    print('')
    print('processing testing set ...')

    test_set = dataset_df[dataset_df['train_only'] == False]  # noqa: E712

    test_images_num = len(test_set)
    meta_info['num_test_images'] = test_images_num
    test_images_per_shard = int(math.ceil(test_images_num / _NUM_SHARDS))
    dataset_dir = meta_info['val_images_dir']
    if dataset_dir is None:
        dataset_dir = meta_info['train_images_dir']

    shard_id = 0
    num_images_seen = 0

    output_filename = os.path.join(
        out_path, '%s-%05d-of-%05d.tfrecord' % ('test', shard_id, _NUM_SHARDS))

    tfrecord_writer = tf.io.TFRecordWriter(output_filename)

    for index, row in test_set.iterrows():

        num_images_seen += 1

        if num_images_seen % test_images_per_shard == 0:
            tfrecord_writer.close()

            shard_id += 1
            output_filename = os.path.join(
                out_path,
                '%s-%05d-of-%05d.tfrecord' % ('test', shard_id, _NUM_SHARDS))

            tfrecord_writer = tf.io.TFRecordWriter(output_filename)

        sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                         (num_images_seen, test_images_num, shard_id))
        sys.stdout.flush()

        label_str = row['annotation_name']
        image_file = os.path.join(dataset_dir, label_str, row['image_name'])

        assert image_file.endswith(
            ('.jpg', '.jpeg', '.png')
        ), 'required `.jpg or .jpeg or .png ` image got instead {}'.format(
            image_file)

        if image_file.endswith('.png'):
            im = Image.open(image_file)
            im.save(_PNG_CONVERT_PATH)
            image_file = _PNG_CONVERT_PATH

        with open(image_file, 'rb') as imgfile:
            image_string = imgfile.read()

        image_array = cv2.imread(image_file)
        image_shape = image_array.shape

        tf_example = image_example(
            image_string=image_string,
            label=class_dict[label_str],
            image_shape=image_shape)

        tfrecord_writer.write(tf_example.SerializeToString())

    tfrecord_writer.close()

    return meta_info


def parse_tfrecords(filenames,
                    height,
                    width,
                    num_classes,
                    processing_func=None,
                    augmentation=None,
                    batch_size=32,
                    num_repeat=-1):

    # if augmentation is not None:
    #   assert isinstance(augmentation, Classification_augmentor)

    _AUGMENTOR = augmentation

    def augment_image(image):
        if _AUGMENTOR is not None:
            return _AUGMENTOR.apply(image=image)
        return image

    # @tf.function
    def decode_cast_resize(image_string):
        image = tf.cast(
            tf.image.decode_jpeg(image_string), tf.keras.backend.floatx())
        image = tf.image.resize(image, size=(height, width))
        image.set_shape([None, None, 3])

        return image

    # @tf.function
    def make_onehot(labels):
        return tf.one_hot(labels, depth=num_classes)

    def _parse_function(serialized):

        features = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image_label': tf.io.FixedLenFeature([], tf.int64)
        }

        parsed_example = tf.io.parse_example(
            serialized=serialized, features=features)

        # image = tf.map_fn(
        #     decode_cast_resize,
        #     parsed_example['image_raw'], dtype=tf.keras.backend.floatx())
        # one_hot_label = tf.map_fn(
        #     make_onehot,
        #     tf.cast(
        #         parsed_example['image_label'], tf.int32),
        #         dtype=tf.keras.backend.floatx())

        # image_height, image_width = parsed_example['image/height'], parsed_example['image/width']  # noqa: E501

        image_batch = []
        label_batch = []

        for batch_index in range(batch_size):

            lbl = parsed_example['image_label'][batch_index]

            image = tf.cast(
                tf.image.decode_jpeg(parsed_example['image_raw'][batch_index]),
                tf.keras.backend.floatx())
            image = tf.image.resize(image, size=(height, width))
            # apply augmentation here
            image = tf.numpy_function(
                func=augment_image,
                inp=[image],
                Tout=tf.keras.backend.floatx())

            if processing_func is not None:
                image = tf.numpy_function(
                    func=processing_func,
                    inp=[image],
                    Tout=tf.keras.backend.floatx())

            image.set_shape([None, None, 3])

            one_hot_label = tf.one_hot(
                tf.cast(lbl, tf.int32), depth=num_classes)

            image_batch.append(image)
            label_batch.append(one_hot_label)

        return tf.convert_to_tensor(image_batch), tf.convert_to_tensor(
            label_batch)

    dataset = tf.data.Dataset.list_files(filenames).shuffle(
        buffer_size=256).repeat(-1)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)  # Batch Size

    dataset = dataset.map(
        _parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
