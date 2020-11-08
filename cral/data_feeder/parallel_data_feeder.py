import concurrent.futures
import json
import os
import tempfile

import cv2
import pandas as pd
import tensorflow as tf
import tqdm
# from cral.augmentations.engine import \
#     Classification as Classification_augmentor
from PIL import Image

_NUM_SHARDS = 4
_PNG_CONVERT_PATH = os.path.join(tempfile.gettempdir(), 'temp.jpg')


def get_label_dict(label_list):
    label_list.sort()
    label_dict = dict()
    for index, label in enumerate(label_list):
        label_dict[label] = index

    return label_dict


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
    """Returns a bytes_list from a list of string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    """Returns a float_list from a list of float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


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
        'image/depth': _int64_feature(image_shape[2]),
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

    # dataset_dir = meta_info['dataset_path']
    class_list = meta_info['class_names']
    class_dict = get_label_dict(class_list)
    label_file_pointer = dict()

    def make_tf_records_parallel(dataset, tfrecord_paths, train=True):

        writers = [tf.io.TFRecordWriter(path) for path in tfrecord_paths]
        total_images = len(dataset)
        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executer:
            for index, row in tqdm.tqdm(
                    dataset.iterrows(), total=total_images, ncols=100):
                if train:
                    label_str = row['annotation_name']
                    if label_str not in label_file_pointer:
                        os.makedirs(
                            os.path.join(out_path, 'statistics'),
                            exist_ok=True)
                        label_file_pointer[label_str] = tf.io.TFRecordWriter(
                            os.path.join(out_path, 'statistics', label_str) +
                            '.tfrecord')

                tasks.append(
                    executer.submit(make_tf_record_row, row,
                                    writers[index % _NUM_SHARDS], train))

                if len(tasks) > _NUM_SHARDS:
                    concurrent.futures.wait(tasks)
                    tasks = []

        concurrent.futures.wait(tasks)
        tasks = []
        for writer in writers:
            writer.close()

        if train:
            for label_stats_pointer_file in label_file_pointer.values():
                label_stats_pointer_file.close()

    def make_tf_record_row(row, writer, train):

        label_str = row['annotation_name']
        image_file = os.path.join(meta_info['dataset_path'], label_str,
                                  row['image_name'])

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

        if train:
            stats_feature = {
                'height': image_shape[0],
                'width': image_shape[1],
            }
            stats_example = statistics_example(data=stats_feature)
            label_file_pointer[label_str].write(
                stats_example.SerializeToString())

        writer.write(tf_example.SerializeToString())

    dataset_df = pd.read_csv(dataset_csv)

    print('\nprocessing training set ...')
    train_set = dataset_df[dataset_df['train_only'] == True]  # noqa: E712
    training_images_num = len(train_set)
    meta_info['num_training_images'] = training_images_num
    tfrecord_paths = []
    for shard_id in range(_NUM_SHARDS):
        tfrecord_paths.append(
            os.path.join(
                out_path, '%s-%05d-of-%05d.tfrecord' %
                ('train', shard_id + 1, _NUM_SHARDS)))

    make_tf_records_parallel(train_set, tfrecord_paths, train=True)

    print('\nprocessing testing set ...')

    test_set = dataset_df[dataset_df['train_only'] == False]  # noqa: E712
    test_images_num = len(test_set)
    meta_info['num_test_images'] = test_images_num
    tfrecord_paths = []
    for shard_id in range(_NUM_SHARDS):
        tfrecord_paths.append(
            os.path.join(
                out_path, '%s-%05d-of-%05d.tfrecord' %
                ('test', shard_id + 1, _NUM_SHARDS)))

    make_tf_records_parallel(test_set, tfrecord_paths, train=False)

    return meta_info
