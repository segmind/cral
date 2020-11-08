import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from cral.data_feeder.utils import _bytes_feature, _int64_feature
from PIL import Image

# number of shard per split
_NUM_SHARDS = 4

# Queue length
_PARALLEL_READS = 16
debug = False


def image_example(image_string, mask_string, image_shape, mask_shape):
    # image_shape = tf.image.decode_jpeg(image_string).shape
    # mask_shape = tf.image.decode_png(mask_string).shape

    feature = {
        'image/height': _int64_feature(image_shape[0]),
        'image/width': _int64_feature(image_shape[1]),
        'image/depth': _int64_feature(image_shape[2]),
        'image_raw': _bytes_feature(image_string),
        'mask/height': _int64_feature(mask_shape[0]),
        'mask/width': _int64_feature(mask_shape[1]),
        'mask/depth': _int64_feature(1),
        'mask_raw': _bytes_feature(mask_string)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _create_tfrecords_from_dataset(image_paths,
                                   image_dir,
                                   mask_dir,
                                   out_dir,
                                   train_only=True):
    '''Args:
    image_paths : List of file-paths for the images
    labels : class-labels for images(vector of size len(image_paths)X1)
    out_path : Destination of TFRecords output file
    size : resize dimensions
    '''

    assert os.path.exists(out_dir), 'directory doesnot exist :: {}'.format(
        out_dir)

    # image_paths = glob.glob(os.path.join(image_dir,'*.jpg'))
    num_images = len(image_paths)
    # print(num_images)
    unique_vals = []

    images_per_shard = num_images // _NUM_SHARDS

    shard_meta = {}
    start_index = 0

    for idx in range(_NUM_SHARDS):
        prefix = 'train' if train_only else 'test'
        end_index = min(start_index + images_per_shard, num_images)
        shard_meta['%s-%05d-of-%05d.tfrecord' %
                   (prefix, idx + 1, _NUM_SHARDS)] = (start_index, end_index)
        start_index = end_index + 1

    for key, (START, END) in shard_meta.items():

        print('Writing :: {}'.format(key))

        with tf.io.TFRecordWriter(os.path.join(out_dir, key)) as writer:

            for image_name in tqdm.tqdm(image_paths[START:END]):
                image_file = os.path.join(image_dir, image_name)

                with open(image_file, 'rb') as imf:
                    image_string = imf.read()
                image_array = np.array(Image.open(image_file))
                image_shape = image_array.shape
                if len(image_shape) != 3:
                    print('ignoring {}, shape : {}'.format(
                        image_file, image_shape))
                    continue

                assert image_shape[
                    2] == 3, f'expected image to have 3 channels but got\
                    {image_shape[2]} instead'

                mask_file = image_file.replace(image_dir, mask_dir).replace(
                    '.jpg', '.png')

                if os.path.isfile(mask_file) is False:  # image has a mask
                    # create an all black mask
                    black_mask = Image.new(
                        mode='L',
                        size=(image_shape[1], image_shape[0]),
                        color=0)
                    # save it as temp.png
                    black_mask.save('temp.png')
                    mask_file = 'temp.png'

                with open(mask_file, 'rb') as mmf:
                    mask_string = mmf.read()
                mask_array = np.array(Image.open(mask_file))
                unique_pixels = np.unique(mask_array).tolist()
                unique_vals = list(set(unique_vals).union(unique_pixels))
                mask_shape = mask_array.shape

                assert len(mask_shape
                           ) == 2, f'expected mask to have 1 channel but got \
                {mask_shape} instead'

                assert mask_shape[0] == image_shape[
                    0], 'mask and image height mismatch'
                assert mask_shape[1] == image_shape[
                    1], 'mask and image width mismatch'

                tf_example = image_example(
                    image_string=image_string,
                    mask_string=mask_string,
                    image_shape=image_shape,
                    mask_shape=mask_shape)

                writer.write(tf_example.SerializeToString())

    print('Num labels in %s-set:: %d' % (prefix, len(unique_vals)))

    return num_images, unique_vals


def create_tfrecords(meta_info,
                     dataset_csv_path,
                     out_path=tempfile.gettempdir()):

    dataset_df = pd.read_csv(dataset_csv_path)
    # with open(meta_json_path,'r') as json_file:
    #     meta_info=json.loads(json_file.read())

    train_dataset = dataset_df[dataset_df['train_only'] ==  # noqa: E712
                               True]['image_name'].tolist()
    train_img_dir = meta_info['train_images_dir']
    train_anno_dir = meta_info['train_anno_dir']

    print('creating tfrecords for training set...')

    meta_info[
        'num_training_images'], num_classes = _create_tfrecords_from_dataset(
            image_paths=train_dataset,
            image_dir=train_img_dir,
            mask_dir=train_anno_dir,
            out_dir=out_path,
            train_only=True)

    test_dataset = dataset_df[dataset_df['train_only'] == False]  # noqa: E712
    if len(test_dataset) > 0:
        test_dataset = test_dataset['image_name'].tolist()
        val_img_dir = meta_info['val_images_dir']
        val_anno_dir = meta_info['val_anno_dir']
        if val_img_dir is None:
            val_img_dir = train_img_dir
        print('creating tfrecords for test set...')
        meta_info[
            'num_test_images'], classes_in_testset = _create_tfrecords_from_dataset(  # noqa: E501
                image_paths=test_dataset,
                image_dir=val_img_dir,
                mask_dir=val_anno_dir,
                out_dir=out_path,
                train_only=False)
        num_classes = list(set(num_classes).union(classes_in_testset))
    else:
        meta_info['num_test_images'] = 0

    meta_info['tfrecord_path'] = out_path
    meta_info['num_classes'] = len(num_classes)

    return meta_info
