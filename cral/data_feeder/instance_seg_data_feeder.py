import tensorflow as tf
import pandas as pd
import tempfile
import json
import os
import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from math import ceil, floor
from PIL import Image
from io import BytesIO
import random
import numpy as np
import matplotlib.pyplot as plt

from cral.data_feeder.utils import (_bytes_feature,
                                    _int64_feature,
                                    _float_list_feature, _int64_list_feature)
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import cv2
# number of shard per split
_NUM_SHARDS = 4

# Queue length
_PARALLEL_READS = 16
debug = False

_PNG_CONVERT_PATH = os.path.join(tempfile.gettempdir(), 'temp')


def _annotate_and_save_image(image_path, xmins, ymins, xmaxs, ymaxs, mask):
    print("DEBUG")
    print(image_path)
#     with Image.open(image_path) as img:
# #         img1=ImageDraw.Draw(img)
#         for i in range(len(xmins)):
#             lst=[xmins[i],ymins[i],xmaxs[i],ymaxs[i]]
# #             img1.rectangle(lst)
# #             print(lst)
#         print(np.unique(mask))
#         print(len(xmins), mask.shape)
    img = cv2.imread(image_path)
    img1 = np.zeros((img.shape[0], img.shape[1]))
    for i in range(mask.shape[2]):
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if(mask[j, k, i]):
                    img1[j, k] = 255
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img1)
#         img.show()
#         sec = input('Let us wait for user input. Let me know how many
#         seconds to sleep now.\n')


def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ast.literal_eval(ann)
#     segm = ast.literal_eval(segm)
#     print(type(segm))
    if isinstance(segm, list):
        # print('if' + str(segm))
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        # print(segm)
        # print(height, width)
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        # print('elif')
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        # print('else')
        rle = ann
        # print('entered')
    return rle


def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or
    RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


def _create_tf_record_row(dataset_group, img_dir):
    xmins = dataset_group['xmin'].tolist()
    ymins = dataset_group['ymin'].tolist()
    xmaxs = dataset_group['xmax'].tolist()
    ymaxs = dataset_group['ymax'].tolist()
    labels = dataset_group['label_id'].tolist()
    allsegs = dataset_group['segmentation'].tolist()
    height = int(dataset_group['image_height'].iloc[0])
    width = int(dataset_group['image_width'].iloc[0])
    instance_masks = []
    for i in range(len(allsegs)):
        m = annToMask(allsegs[i], int(dataset_group['image_height'].iloc[0]),
                      int(dataset_group['image_width'].iloc[0]))
        instance_masks.append(m)
    mask = np.stack(instance_masks, axis=2).astype(np.bool)
    mask = np.array(mask)
    mask_bytes = mask.tobytes()
    image_shape = None  # noqa: F841
    image_string = None

    image_file = os.path.join(img_dir, dataset_group['image_name'].iloc[0])
    if random.random() > 0.9 and debug:
        _annotate_and_save_image(image_file, xmins, ymins, xmaxs, ymaxs, mask)
    image_id = int(dataset_group['image_id'].iloc[0])
    height = int(dataset_group['image_height'].iloc[0])
    width = int(dataset_group['image_width'].iloc[0])

    assert image_file.endswith(('.jpg', '.jpeg', '.png')), 'required `.jpg or .jpeg or .png ` image got instead {}'.format(image_file)  # noqa:E501
    if image_file.endswith('.png'):
        img = Image.open(image_file + str())
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image_file = BytesIO()
        img.save(image_file, format='jpeg')

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
        # image_shape = img.size
    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width),
          'image/encoded': _bytes_feature(image_string),
          'image/object/bbox/xmin': _float_list_feature(xmins),
          'image/object/bbox/xmax': _float_list_feature(xmaxs),
          'image/object/bbox/ymin': _float_list_feature(ymins),
          'image/object/bbox/ymax': _float_list_feature(ymaxs),
          'image/f_id': _int64_feature(image_id),
          'image/object/class/label': _int64_list_feature(labels),
          'image/object/mask': _bytes_feature(mask_bytes),
      }))
    return [tf_example.SerializeToString()]


def _create_tfrecords_from_dataset(dataset, img_dir, out_path, train_only=True):  # noqa:E501

    # Get size of train/test set
    dataset_groups = [df for _, df in dataset.groupby('image_id')]
    total_images = len(dataset_groups)

    # decide on number of images to write per shard
    num_images_per_shard = ceil(total_images/_NUM_SHARDS)

    subset = 'test'
    if train_only:
        subset = 'train'
    print(f'\nCreating {subset} Tfrecords')

    tfrecord_paths = []
    for shard_id in range(_NUM_SHARDS):
        tfrecord_paths.append(
          os.path.join(
            out_path, '%s-%05d-of-%05d.tfrecord' %
            (subset, shard_id+1, _NUM_SHARDS)))

    # initialize tfrecord writer for each shard
    writers = [tf.io.TFRecordWriter(path) for path in tfrecord_paths]

    dataset_groups_list = []
    indexs = []
    index = 0
    for dataset_group in tqdm.tqdm(dataset_groups, ncols=100):
        # result=_create_tf_record_row(dataset_group, img_dir, index=1)
        # writers[0].write(result)

        # populate list(queue) with images and annotation instance
        dataset_groups_list.append(dataset_group)
        indexs.append(index)
        index += 1

        # launch threads when max limit of list(Queue) is reached
        if len(dataset_groups_list) == _PARALLEL_READS:
            with ThreadPoolExecutor() as executer:
                # fire threads, which return serialized example for each image
                # and it's annotation
                results = executer.map(_create_tf_record_row,
                                       dataset_groups_list, repeat(img_dir,
                                                                   len(dataset_groups_list)))  # noqa: E501

                # write-out the serialized example to it's respective shard
                for i, result in enumerate(results):
                    writer = writers[floor(indexs[i]/num_images_per_shard)]
                    writer.write(result[0])
                results = []
                dataset_groups_list = []
                indexs = []

    # for the last case when queue is not totally full
    if len(dataset_groups_list) > 0:
        with ThreadPoolExecutor() as executer:
            results = executer.map(_create_tf_record_row, dataset_groups_list,
                                   repeat(img_dir, len(dataset_groups_list)))
            for i, result in enumerate(results):
                writer = writers[floor(indexs[i]/num_images_per_shard)]
                writer.write(result[0])
            results = []
            dataset_groups_list = []
            indexs = []

    # close each tfrecord writer
    for writer in writers:
        writer.close()

    return total_images


def create_tfrecords(meta_json_path, dataset_csv_path, out_path='/tmp/'):

    dataset_df = pd.read_csv(dataset_csv_path)
    with open(meta_json_path, 'r') as json_file:
        meta_info = json.loads(json_file.read())
#     print(meta_info)
    train_dataset = dataset_df[dataset_df['train_only'] == True]  # noqa: E712
    train_img_dir = meta_info['train_image_dir']
    meta_info['num_training_images'] = _create_tfrecords_from_dataset(
      train_dataset, train_img_dir, train_only=True, out_path=out_path)

    train_dataset = dataset_df[dataset_df['train_only'] == False]  # noqa: E712
    if len(train_dataset) > 0:
        val_img_dir = meta_info['val_image_dir']
        if val_img_dir is None:
            val_img_dir = train_img_dir
        meta_info['num_test_images'] = _create_tfrecords_from_dataset(
          train_dataset, val_img_dir, train_only=False, out_path=out_path)
    else:
        meta_info['num_test_images'] = 0

    meta_info['tfrecord_path'] = out_path
#     print(meta_info)
    return meta_info

# csv_path="C:\\Users\\yadur\\AppData\\Local\\Temp\\dataset.csv"
# json_path="C:\\Users\\yadur\\AppData\\Local\\Temp\\dataset.json"
# create_tfrecords(json_path,csv_path)
