import concurrent.futures
import json
import os
import tempfile
from io import BytesIO
from itertools import repeat
from math import ceil, floor

import pandas as pd
import tensorflow as tf
import tqdm
from cral.data_feeder.utils import _bytes_feature, _int64_feature
from PIL import Image

_NUM_SHARDS = 4
_PARALLEL_READS = 16
_PNG_CONVERT_PATH = os.path.join(tempfile.gettempdir(), 'temp')


def get_label_dict(label_list):
    label_list.sort()
    label_dict = dict()
    for index, label in enumerate(label_list):
        label_dict[label] = index

    return label_dict


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

    with open(meta_json, 'r') as f:
        meta_info = json.loads(f.read())

    meta_info['num_training_images'] = None
    meta_info['num_test_images'] = None
    meta_info['tfrecord_path'] = out_path

    # dataset_dir = meta_info['dataset_path']

    class_list = meta_info['class_names']
    class_dict = get_label_dict(class_list)
    label_file_pointer = dict()

    def make_tf_records_parallel(dataset, tfrecord_paths, img_dir, train=True):

        writers = [tf.io.TFRecordWriter(path) for path in tfrecord_paths]
        total_images = len(dataset)
        num_seen_images = 0
        num_images_per_shard = ceil(total_images / _NUM_SHARDS)
        rows = []
        indexs = []
        for index, row in tqdm.tqdm(
                dataset.iterrows(), total=total_images, ncols=100):
            if train:
                label_str = row['annotation_name']
                if label_str not in label_file_pointer:
                    os.makedirs(
                        os.path.join(out_path, 'statistics'), exist_ok=True)
                    label_file_pointer[label_str] = tf.io.TFRecordWriter(
                        os.path.join(out_path, 'statistics', label_str) +
                        '.tfrecord')
            rows.append(row)
            indexs.append(num_seen_images)
            num_seen_images += 1
            if len(rows) == _PARALLEL_READS:
                with concurrent.futures.ThreadPoolExecutor() as executer:
                    results = executer.map(make_tf_record_row, rows,
                                           repeat(train, len(rows)),
                                           repeat(img_dir, len(rows)))
                for i, result in enumerate(results):
                    if result is not None:
                        writer = writers[floor(indexs[i] /
                                               num_images_per_shard)]
                        writer.write(result[0])
                        if train:
                            label_file_pointer[result[1]].write(result[2])
                rows = []
                indexs = []

        if len(rows) > 0:
            with concurrent.futures.ThreadPoolExecutor() as executer:
                results = executer.map(make_tf_record_row, rows,
                                       repeat(train, len(rows)),
                                       repeat(img_dir, len(rows)))
            for i, result in enumerate(results):
                if result is not None:
                    writer = writers[floor(indexs[i] / num_images_per_shard)]
                    writer.write(result[0])
                    if train:
                        label_file_pointer[result[1]].write(result[2])
            rows = []
            indexs = []

        for writer in writers:
            writer.close()

        if train:
            for label_stats_pointer_file in label_file_pointer.values():
                label_stats_pointer_file.close()

    def make_tf_record_row(row, train, img_dir):
        label_str = row['annotation_name']
        image_file = os.path.join(img_dir, label_str, row['image_name'])
        image_shape = None
        image_string = None
        assert image_file.endswith(
            ('.jpg', '.jpeg', '.png')
        ), 'required `.jpg or .jpeg or .png ` image got instead {}'.format(
            image_file)
        if image_file.endswith('.png'):
            im = Image.open(image_file + str())
            image_file = BytesIO()
            im.save(image_file, format='jpeg')

        with Image.open(image_file) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                byts = BytesIO()
                img.save(byts, format='jpeg')
                image_string = byts.getvalue()
            else:
                if type(image_file) is str:
                    with open(image_file, 'rb') as imgfile:
                        image_string = imgfile.read()
                else:
                    image_string = image_file.getvalue()
            image_shape = img.size

        if image_shape is None or image_string is None:
            print(f"Not able to load {row['image_name']}")
            return None

        tf_example = image_example(
            image_string=image_string,
            label=class_dict[label_str],
            image_shape=image_shape)

        return [tf_example.SerializeToString()]

    dataset_df = pd.read_csv(dataset_csv)
    print('\nprocessing training set ...')
    train_set = dataset_df[dataset_df['train_only'] == True]  # noqa: E712
    training_images_num = len(train_set)
    meta_info['num_training_images'] = training_images_num
    image_dir = meta_info['train_images_dir']
    tfrecord_paths = []
    for shard_id in range(_NUM_SHARDS):
        tfrecord_paths.append(
            os.path.join(
                out_path, '%s-%05d-of-%05d.tfrecord' %
                ('train', shard_id + 1, _NUM_SHARDS)))

    make_tf_records_parallel(train_set, tfrecord_paths, image_dir, train=True)

    test_set = dataset_df[dataset_df['train_only'] == False]  # noqa: E712
    test_images_num = len(test_set)
    meta_info['num_test_images'] = test_images_num
    tfrecord_paths = []
    image_dir = meta_info['val_images_dir']
    if image_dir is None:
        image_dir = meta_info['train_images_dir']
    if test_images_num > 0:
        print('\nprocessing testing set ...')
        for shard_id in range(_NUM_SHARDS):
            tfrecord_paths.append(
                os.path.join(
                    out_path, '%s-%05d-of-%05d.tfrecord' %
                    ('test', shard_id + 1, _NUM_SHARDS)))

        make_tf_records_parallel(
            test_set, tfrecord_paths, image_dir, train=False)

    return meta_info
