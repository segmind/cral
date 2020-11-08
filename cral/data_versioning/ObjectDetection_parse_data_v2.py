import json
import os
import random
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import pandas as pd
import tqdm
from cral.data_versioning.cral_hash import hashFile
from cral.data_versioning.cral_util import fileName, find_images
from PIL import Image

_ALLOWED_ANNOTATION_FORMATS = ('coco', 'pascal_voc', 'yolo')
_EXT = {'pascal_voc': '.xml', 'yolo': '.txt'}
_PARALLEL_READS = 16
_RANDOM_SEED = 12


def find_classes(dataset):
    return sorted(dataset['label_name'].unique().tolist())


def set_up_label_ids(dataset):
    _CLASSES = find_classes(dataset)
    label_to_id = {}
    if 'background_empty' in _CLASSES:
        label_to_id['background_empty'] = -1
        _CLASSES.remove('background_empty')
    for i, name in enumerate(_CLASSES):
        label_to_id[name] = i
    dataset['label_id'] = dataset['label_name'].map(label_to_id)
    return dataset


def give_dataset_dict(dataset, train_images_dir, train_anno_dir,
                      val_images_dir, val_anno_dir):
    _CLASSES = find_classes(dataset)
    if 'background_empty' in _CLASSES:
        _CLASSES.remove('background_empty')
    meta_info = {
        'task_type': 'object_detection',
        'train_image_dir': train_images_dir,
        'train_anno_dir': train_anno_dir,
        'val_image_dir': val_images_dir,
        'val_anno_dir': val_anno_dir,
        'num_classes': len(_CLASSES),
        'classes': _CLASSES
    }
    return meta_info


def get_datapoint_pascal(image_path, anno_path):
    dataset_info = {
        'image_name': [],
        'image_hash': [],
        'image_width': [],
        'image_height': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'label_name': []
    }
    num_annos = 0
    _hash = hashFile(image_path)
    with Image.open(image_path) as im:
        image_shape = im.size

    if anno_path is not None:
        file = ET.parse(anno_path)
        for anno in file.iter('object'):
            dataset_info['xmin'].append(
                float(anno.find('bndbox').find('xmin').text))
            dataset_info['ymin'].append(
                float(anno.find('bndbox').find('ymin').text))
            dataset_info['xmax'].append(
                float(anno.find('bndbox').find('xmax').text))
            dataset_info['ymax'].append(
                float(anno.find('bndbox').find('ymax').text))
            dataset_info['label_name'].append(anno.find('name').text)
            num_annos += 1
    else:
        num_annos = 1
        dataset_info['xmin'].append(-1.0)
        dataset_info['ymin'].append(-1.0)
        dataset_info['xmax'].append(-1.0)
        dataset_info['ymax'].append(-1.0)
        dataset_info['label_name'].append('background_empty')

    dataset_info['image_name'].extend(
        repeat(fileName(image_path, ext=True), num_annos))
    dataset_info['image_width'].extend(repeat(image_shape[0], num_annos))
    dataset_info['image_height'].extend(repeat(image_shape[1], num_annos))
    dataset_info['image_hash'].extend(repeat(_hash, num_annos))
    return dataset_info


def map_result_to_data(results, dataset_info):
    for data in results:
        dataset_info['image_name'].extend(data['image_name'])
        dataset_info['image_hash'].extend(data['image_hash'])
        dataset_info['image_width'].extend(data['image_width'])
        dataset_info['image_height'].extend(data['image_height'])
        dataset_info['xmin'].extend(data['xmin'])
        dataset_info['ymin'].extend(data['ymin'])
        dataset_info['xmax'].extend(data['xmax'])
        dataset_info['ymax'].extend(data['ymax'])
        dataset_info['label_name'].extend(data['label_name'])


def get_dataset_info_pascal(image_dir,
                            anno_dir,
                            img_to_anno=None,
                            start_id=0,
                            split=None,
                            train_only=True,
                            *args,
                            **kwargs):
    images = find_images(image_dir)
    dataset_info = {
        'image_id': [],
        'image_name': [],
        'image_hash': [],
        'image_width': [],
        'image_height': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'label_name': [],
        'train_only': []
    }
    image_paths = []
    anno_paths = []
    subset = 'Test'
    if train_only:
        subset = 'Training'
    print(f'\n\nProcessing {subset} Dataset')
    for curr_image_path in tqdm.tqdm(images, ncols=100):
        if img_to_anno is not None:
            curr_anno_name = img_to_anno(fileName(curr_image_path))
        else:
            curr_anno_name = fileName(curr_image_path)
        curr_anno_path = os.path.join(anno_dir, curr_anno_name + '.xml')
        image_paths.append(curr_image_path)
        if os.path.isfile(curr_anno_path):
            anno_paths.append(curr_anno_path)
        else:
            anno_paths.append(None)
        if len(image_paths) == _PARALLEL_READS:
            with ThreadPoolExecutor() as executer:
                results = executer.map(get_datapoint_pascal, image_paths,
                                       anno_paths)
            map_result_to_data(results, dataset_info)
            image_paths = []
            anno_paths = []
    if len(image_paths) > 0:
        with ThreadPoolExecutor() as executer:
            results = executer.map(get_datapoint_pascal, image_paths,
                                   anno_paths)
        map_result_to_data(results, dataset_info)
    dataset_info['train_only'] = repeat(train_only,
                                        len(dataset_info['image_name']))
    dataset_info['image_id'] = repeat(0, len(dataset_info['image_name']))
    dataset_df = pd.DataFrame.from_dict(dataset_info)
    grouped = [df for _, df in dataset_df.groupby('image_name')]
    total_images = len(grouped)
    random.shuffle(grouped)
    dataset_df = pd.concat(grouped).reset_index(drop=True)
    dataset_df['image_id'] = dataset_df.groupby(
        'image_name', sort=False).ngroup() + start_id
    if split != 0 and split is not None:
        split_len = split * total_images
        dataset_df.loc[dataset_df['image_id'] < start_id + split_len,
                       'train_only'] = False
    return dataset_df


def get_datapoint_yolo(image_path, anno_path, _CLASSES):
    dataset_info = {
        'image_name': [],
        'image_hash': [],
        'image_width': [],
        'image_height': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'label_name': []
    }
    file = open(anno_path)
    num_annos = 0
    with Image.open(image_path) as im:
        image_shape = im.size
    _hash = hashFile(image_path)
    if anno_path is not None:
        for anno in file:
            anno = anno.split(' ')
            anno[0] = int(anno[0])
            anno[1] = float(anno[1])
            anno[2] = float(anno[2])
            anno[3] = float(anno[3])
            anno[4] = float(anno[4])
            label = anno[0]
            xmin = image_shape[0] * (anno[1] - anno[3] / 2.0)
            ymin = image_shape[1] * (anno[2] - anno[4] / 2.0)
            xmax = image_shape[0] * (anno[1] + anno[3] / 2.0)
            ymax = image_shape[1] * (anno[2] + anno[4] / 2.0)
            dataset_info['xmin'].append(xmin)
            dataset_info['ymin'].append(ymin)
            dataset_info['xmax'].append(xmax)
            dataset_info['ymax'].append(ymax)
            if len(_CLASSES) != 0:
                dataset_info['label_name'].append(_CLASSES[label])
            else:
                dataset_info['label_name'].append(label)
            num_annos += 1
    else:
        num_annos = 1
        dataset_info['xmin'].append(-1.0)
        dataset_info['ymin'].append(-1.0)
        dataset_info['xmax'].append(-1.0)
        dataset_info['ymax'].append(-1.0)
        dataset_info['label_name'].append('background_empty')

    dataset_info['image_name'].extend(
        repeat(fileName(image_path, ext=True), num_annos))
    dataset_info['image_width'].extend(repeat(image_shape[0], num_annos))
    dataset_info['image_height'].extend(repeat(image_shape[1], num_annos))
    dataset_info['image_hash'].extend(repeat(_hash, num_annos))
    return dataset_info


def get_dataset_info_yolo(image_dir,
                          anno_dir,
                          names_file=None,
                          img_to_anno=None,
                          start_id=0,
                          split=None,
                          train_only=True,
                          *args,
                          **kwargs):
    _CLASSES = []
    if names_file is not None:
        with open(names_file) as file:
            _CLASSES = [x[:-1] for x in file]

    images = find_images(image_dir)
    dataset_info = {
        'image_id': [],
        'image_name': [],
        'image_hash': [],
        'image_width': [],
        'image_height': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'label_name': [],
        'train_only': []
    }

    image_paths = []
    anno_paths = []
    # anno_counts = []
    subset = 'Test'
    if train_only:
        subset = 'Training'
    print(f'\nProcessing {subset} Dataset')

    for curr_image_path in tqdm.tqdm(images, ncols=100):
        if img_to_anno is not None:
            curr_anno_name = img_to_anno(fileName(curr_image_path))
        else:
            curr_anno_name = fileName(curr_image_path)
        curr_anno_path = os.path.join(anno_dir, curr_anno_name + '.txt')
        image_paths.append(curr_image_path)
        if os.path.isfile(curr_anno_path):
            anno_paths.append(curr_anno_path)
        else:
            anno_paths.append(None)
        if len(image_paths) == _PARALLEL_READS:
            with ThreadPoolExecutor() as executer:
                results = executer.map(get_datapoint_yolo, image_paths,
                                       anno_paths, repeat(_CLASSES))
            map_result_to_data(results, dataset_info)
            image_paths = []
            anno_paths = []
    if len(image_paths) > 0:
        with ThreadPoolExecutor() as executer:
            results = executer.map(get_datapoint_yolo, image_paths, anno_paths,
                                   repeat(_CLASSES))
        map_result_to_data(results, dataset_info)
        image_paths = []
        anno_paths = []
    dataset_info['train_only'] = repeat(train_only,
                                        len(dataset_info['image_name']))
    dataset_info['image_id'] = repeat(0, len(dataset_info['image_name']))
    dataset_df = pd.DataFrame.from_dict(dataset_info)
    grouped = [df for _, df in dataset_df.groupby('image_name')]
    total_images = len(grouped)
    random.shuffle(grouped)
    dataset_df = pd.concat(grouped).reset_index(drop=True)
    dataset_df['image_id'] = dataset_df.groupby(
        'image_name', sort=False).ngroup() + start_id
    if split != 0 and split is not None:
        split_len = split * total_images
        dataset_df.loc[dataset_df['image_id'] < start_id + split_len,
                       'train_only'] = False
    return dataset_df


def coco_image_info(image_info):
    image_info['image_hash'] = hashFile(image_info['image_path'])


def give_coco_datapoint(image, anno, dataset_info, cat_id_to_cat):
    xmin = anno['bbox'][0]
    ymin = anno['bbox'][1]
    xmax = anno['bbox'][0] + anno['bbox'][2]
    ymax = anno['bbox'][1] + anno['bbox'][3]
    label_name = cat_id_to_cat[str(anno['category_id'])]
    dataset_info['image_id'].append(image['image_id'])
    dataset_info['image_name'].append(image['image_name'])
    dataset_info['image_hash'].append(image['image_hash'])
    dataset_info['image_height'].append(image['image_height'])
    dataset_info['image_width'].append(image['image_width'])
    dataset_info['xmin'].append(xmin)
    dataset_info['ymin'].append(ymin)
    dataset_info['xmax'].append(xmax)
    dataset_info['ymax'].append(ymax)
    dataset_info['label_name'].append(label_name)


def get_dataset_info_coco(image_dir,
                          anno_dir,
                          start_id=0,
                          split=None,
                          train_only=True,
                          *args,
                          **kwargs):
    with open(anno_dir) as anno_file:
        anno_file = json.load(anno_file)
    # print(anno_file['images'][0])
    # print(anno_file['annotations'][0])
    id_to_image = {}
    image_info_list = []
    image_num = 0
    subset = 'Test'
    if train_only:
        subset = 'Training'
    print(f'\nFinding {subset} images')
    for image in tqdm.tqdm(anno_file['images'], ncols=100):
        image_name = image['file_name']
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            image_info = {
                'image_id': 0,
                'image_name': image_name,
                'image_hash': None,
                'image_height': image['height'],
                'image_width': image['width'],
                'orignal_id': str(image['id']),
                'image_path': image_path
            }
            image_info_list.append(image_info)
            image_num += 1
    print(f'\nProcessing {subset} Images')
    with ThreadPoolExecutor() as executer:
        results = list(  # noqa: F841
            tqdm.tqdm(
                executer.map(coco_image_info, image_info_list),
                total=len(image_info_list),
                ncols=100))

    for image in image_info_list:
        id_to_image[str(image['orignal_id'])] = image

    dataset_info = {
        'image_id': [],
        'image_name': [],
        'image_hash': [],
        'image_width': [],
        'image_height': [],
        'xmin': [],
        'ymin': [],
        'xmax': [],
        'ymax': [],
        'label_name': [],
        'train_only': []
    }

    print(f'\nProcessing {subset} Annotations')
    cat_id_to_cat = {}
    for cat in anno_file['categories']:
        cat_id_to_cat[str(cat['id'])] = cat['name']

    for anno in tqdm.tqdm(anno_file['annotations'], ncols=100):

        orignal_id = str(anno['image_id'])

        if orignal_id in id_to_image:
            image = id_to_image[orignal_id]
            give_coco_datapoint(image, anno, dataset_info, cat_id_to_cat)

    dataset_info['train_only'] = repeat(train_only,
                                        len(dataset_info['image_name']))
    dataset_info['image_id'] = repeat(0, len(dataset_info['image_name']))
    dataset_df = pd.DataFrame.from_dict(dataset_info)
    grouped = [df for _, df in dataset_df.groupby('image_name')]
    random.shuffle(grouped)
    total_images = len(grouped)
    dataset_df = pd.concat(grouped).reset_index(drop=True)
    dataset_df['image_id'] = dataset_df.groupby(
        'image_name', sort=False).ngroup() + start_id
    if split != 0 and split is not None:
        split_len = split * total_images
        dataset_df.loc[dataset_df['image_id'] < start_id + split_len,
                       'train_only'] = False

    return dataset_df


def make_csv(annotation_format,
             train_images_dir,
             train_anno_dir,
             csv_dir=tempfile.gettempdir(),
             val_images_dir=None,
             val_anno_dir=None,
             names_file=None,
             split=None,
             img_to_anno=None):
    """Parses the data and makes a csv file and returns its hash.

    Args:
        annotation_format (str): one of "yolo","coco","pascal_voc"
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        csv_dir (str): path to save the CSV file created
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        names_file (None, optional): Path to .names file in YOLO format
        split (float, optional): float to divide training dataset into
                training and val
        img_to_anno (function, optional): Function to convert image name to
                annotation name

    Returns:
        str: Hash of the csv file created
    """

    assert os.path.isdir(csv_dir), f'{csv_dir} is not a directory'
    assert os.path.isdir(
        train_images_dir), f'{train_images_dir} is not a directory'
    assert isinstance(
        annotation_format, str
    ), f'annotation_format has to be of type str but got {type(annotation_format)} instead'  # noqa: E501
    annotation_format = annotation_format.lower()
    assert annotation_format in _ALLOWED_ANNOTATION_FORMATS, \
        f'supported annotation formats are {_ALLOWED_ANNOTATION_FORMATS}'
    if annotation_format == 'coco':
        assert os.path.isfile(train_anno_dir) and train_anno_dir.endswith(
            '.json'), f'{train_anno_dir} is not a json file'
    else:
        assert os.path.isdir(
            train_anno_dir), f'{train_anno_dir} is not a directory'

    get_dataset_info = None
    random.seed(_RANDOM_SEED)

    if annotation_format == 'coco':
        get_dataset_info = get_dataset_info_coco
    elif annotation_format == 'yolo':
        get_dataset_info = get_dataset_info_yolo
    elif annotation_format == 'pascal_voc':
        get_dataset_info = get_dataset_info_pascal

    if val_images_dir or val_anno_dir:
        assert os.path.isdir(
            val_images_dir), f'{val_images_dir} is not a directory'
        if annotation_format == 'coco':
            assert os.path.isfile(val_anno_dir) and val_anno_dir.endswith(
                '.json'), f'{val_anno_dir} is not a json file'
        else:
            assert os.path.isdir(
                val_anno_dir), f'{val_anno_dir} is not a directory'

        train_dataset_df = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            names_file=names_file,
            img_to_anno=img_to_anno)
        total_images = len(train_dataset_df.groupby('image_name'))
        val_dataset_df = get_dataset_info(
            image_dir=val_images_dir,
            anno_dir=val_anno_dir,
            names_file=names_file,
            img_to_anno=img_to_anno,
            start_id=total_images,
            train_only=False)
        dataset_df = pd.concat([train_dataset_df, val_dataset_df])

    elif split is not None:
        assert isinstance(split, float) or isinstance(
            split, int), f'expected to be float, but got {type(split)} instead'
        assert 0 <= split <= 1.0, f'expected a float between 0 and 1, but got {split} instead'  # noqa: E501
        dataset_df = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            names_file=names_file,
            img_to_anno=img_to_anno,
            split=split)
    else:
        dataset_df = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            names_file=names_file,
            img_to_anno=img_to_anno)

    dataset_df = set_up_label_ids(dataset_df)

    csv_save_path = os.path.join(csv_dir, 'dataset.csv')
    dataset_df.to_csv(csv_save_path, index=False)

    json_save_path = os.path.join(csv_dir, 'dataset.json')
    meta_into = give_dataset_dict(dataset_df, train_images_dir, train_anno_dir,
                                  val_images_dir, val_anno_dir)
    with open(json_save_path, 'w') as json_file:
        json.dump(meta_into, json_file)

    return hashFile(csv_save_path), csv_save_path, json_save_path
