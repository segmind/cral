import datetime
import os
import random
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import numpy as np
import tensorflow as tf
import tqdm
from cral.data_versioning.cral_hash import hashFile
from cral.data_versioning.cral_util import fileName, find_images
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

_ALLOWED_ANNOTATION_FORMATS = ('pascal', 'yolo')
_PARALLEL_READS = 16
_RANDOM_SEED = 12

INFO = {
    'description': 'Example Dataset',
    'url': 'https://github.com/waspinator/pycococreator',
    'version': '0.1.0',
    'year': 2018,
    'contributor': 'waspinator',
    'date_created': datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [{
    'id': 1,
    'name': 'Attribution-NonCommercial-ShareAlike License',
    'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'
}]

font = ImageFont.load_default()

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
    'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
    'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick',
    'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
    'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory',
    'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon',
    'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
    'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen',
    'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise',
    'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite',
    'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid',
    'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
    'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat',
    'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def get_classes_ids(annos):
    _CLASSES = []
    for anno in annos:
        if anno['category_id'] not in _CLASSES:
            if anno['category_id'].isnumeric():
                _CLASSES.append(int(anno['category_id']))
            else:
                _CLASSES.append(anno['category_id'])
    _CLASSES = sorted(_CLASSES)
    _CLASSES = [str(x) for x in _CLASSES]
    catagories = []
    label_to_id = {}
    for index, label in enumerate(_CLASSES):
        catagories.append({
            'id': index,
            'name': label,
            'supercategory': 'custom'
        })
        label_to_id[label] = index
    return catagories, label_to_id


def set_category_id(anno, label_to_id):
    anno['category_id'] = label_to_id[anno['category_id']]


def make_coco_dataset(image_with_anno, split=None):
    if split != 0 and split is not None:

        split_len = round(split * len(image_with_anno))
        train_data = image_with_anno[split_len:]
        val_data = image_with_anno[:split_len]
        return make_coco_dataset(train_data), make_coco_dataset(val_data)
    else:
        images, annos = list(zip(*image_with_anno))
        annos_temp = []
        for anno_grp in annos:
            annos_temp.extend(anno_grp)
        annos = annos_temp
        CATEGORIES, label_to_id = get_classes_ids(annos)

        with ThreadPoolExecutor() as executer:
            _ = executer.map(set_category_id, annos, repeat(label_to_id))

    return {
        'info': INFO,
        'licenses': LICENSES,
        'categories': CATEGORIES,
        'images': images,
        'annotations': annos
    }


def give_anno_info_pascal_coco(anno, image_id, anno_id):
    xmin = float(anno.find('bndbox').find('xmin').text)
    ymin = float(anno.find('bndbox').find('ymin').text)
    xmax = float(anno.find('bndbox').find('xmax').text)
    ymax = float(anno.find('bndbox').find('ymax').text)
    o_width = xmax - xmin
    o_height = ymax - ymin
    label = anno.find('name').text
    return {
        'id': anno_id,
        'image_id': image_id,
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': label,
        'segmentation': []
    }


def give_image_info_coco(
        image_path,
        image_id,
        date_captured=datetime.datetime.utcnow().isoformat(' '),
        license_id=1,
        coco_url='',
        flickr_url=''):
    with Image.open(image_path) as im:
        image_size = im.size

    return {
        'id': image_id,
        'file_name': fileName(image_path, ext=True),
        'width': image_size[0],
        'height': image_size[1],
        'date_captured': date_captured,
        'license': license_id,
        'coco_url': coco_url,
        'flickr_url': flickr_url,
        'segmind_image_hash': hashFile(image_path)
    }


def get_dataset_info_pascal(image_dir, anno_dir, img_to_anno=None, split=None):
    images = find_images(image_dir)
    info_images = []
    info_annotations = []
    image_num = 0
    anno_num = 0
    image_paths = []
    images_ids = []
    for curr_image_path in tqdm.tqdm(images):

        if img_to_anno is not None:
            curr_anno_name = img_to_anno(fileName(curr_image_path))
        else:
            curr_anno_name = fileName(curr_image_path)

        curr_anno_path = os.path.join(anno_dir, curr_anno_name + '.xml')
        if os.path.isfile(curr_anno_path):
            image_paths.append(curr_image_path)
            images_ids.append(image_num)
            file = ET.parse(curr_anno_path)
            curr_anno_info = []
            for anno in file.iter('object'):
                curr_anno_info.append(
                    give_anno_info_pascal_coco(anno, image_num, anno_num))
                anno_num += 1
            image_num += 1
            info_annotations.append(curr_anno_info)

            if len(image_paths) == _PARALLEL_READS:
                with ThreadPoolExecutor() as executer:
                    results = executer.map(give_image_info_coco, image_paths,
                                           images_ids)
                info_images.extend(results)
                image_paths = []
                images_ids = []

    if len(image_paths) > 0:
        with ThreadPoolExecutor() as executer:
            results = executer.map(give_image_info_coco, image_paths,
                                   images_ids)
        info_images.extend(results)
        image_paths = []
        images_ids = []

    image_with_anno = list(zip(info_images, info_annotations))
    random.shuffle(image_with_anno)
    return make_coco_dataset(image_with_anno, split)


def give_anno_info_single_yolo_coco(anno, image_shape, image_id, anno_id):
    anno = anno.split(' ')
    anno[1] = float(anno[1])
    anno[2] = float(anno[2])
    anno[3] = float(anno[3])
    anno[4] = float(anno[4])
    label_id = anno[0]
    xmin = image_shape[0] * (anno[1] - anno[3] / 2.0)
    ymin = image_shape[1] * (anno[2] - anno[4] / 2.0)
    o_width = image_shape[0] * anno[3]
    o_height = image_shape[1] * anno[4]
    xmin = xmin
    ymin = ymin
    o_width = o_width
    o_height = o_height
    return {
        'id': anno_id,
        'image_id': image_id,
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': label_id,
        'segmentation': []
    }


def give_anno_info_batch_yolo_coco(anno_path, image_path, image_id, anno_id):
    annnotations = []
    im = Image.open(image_path)
    image_shape = im.size
    with open(anno_path) as anno_file:
        for i, anno in enumerate(anno_file):
            annnotations.append(
                give_anno_info_single_yolo_coco(anno, image_shape, image_id,
                                                anno_id + i))
    return annnotations


def get_dataset_info_yolo(image_dir, anno_dir, img_to_anno=None, split=None):
    images = find_images(image_dir)
    info_images = []
    info_annotations = []
    image_num = 0
    anno_num = 0
    image_paths = []
    images_ids = []
    for curr_image_path in tqdm.tqdm(images):
        image_name = fileName(curr_image_path)
        if img_to_anno is not None:
            curr_anno_name = img_to_anno(image_name)
        else:
            curr_anno_name = image_name

        curr_anno_path = os.path.join(anno_dir, curr_anno_name + '.txt')
        if os.path.isfile(curr_anno_path):
            image_paths.append(curr_image_path)
            images_ids.append(image_num)
            info_annotations.append(
                give_anno_info_batch_yolo_coco(curr_anno_path, curr_image_path,
                                               image_num, anno_num))
            anno_num += len(info_annotations[-1])
            image_num += 1

            if len(image_paths) == _PARALLEL_READS:
                with ThreadPoolExecutor() as executer:
                    results = executer.map(give_image_info_coco, image_paths,
                                           images_ids)
                info_images.extend(results)
                image_paths = []
                images_ids = []

    if len(image_paths) > 0:
        with ThreadPoolExecutor() as executer:
            results = executer.map(give_image_info_coco, image_paths,
                                   images_ids)
        info_images.extend(results)
        image_paths = []
        images_ids = []

    image_with_anno = list(zip(info_images, info_annotations))
    # random.shuffle(image_with_anno)
    return make_coco_dataset(image_with_anno, split)


def convert_to_coco(annotation_format,
                    train_images_dir,
                    train_anno_dir,
                    val_images_dir=None,
                    val_anno_dir=None,
                    split=None,
                    img_to_anno=None):
    """Converts Dataset to coco Format
    Args:
        annotation_format (str): one of "yolo","pascal"
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        split (float, optional): float to divide training dataset into
                                training and val
        img_to_anno (function, optional): Function to convert image name to
                                annotation name

    Returns:
        str: dict with dataset info in coco format
    """
    # assert os.path.isdir(csv_dir), f"{csv_dir} is not a directory"
    assert os.path.isdir(
        train_images_dir), f'{train_images_dir} is not a directory'
    assert isinstance(annotation_format,
                      str), f'annotation_format has to be of type str but got \
    {type(annotation_format)} instead'

    annotation_format = annotation_format.lower()
    assert annotation_format in _ALLOWED_ANNOTATION_FORMATS, f'supported\
    annotation formats are {_ALLOWED_ANNOTATION_FORMATS}'

    if annotation_format == 'coco':
        assert os.path.isfile(train_anno_dir) and train_anno_dir.endswith(
            '.json'), f'{train_anno_dir} is not a json file'
    else:
        assert os.path.isdir(
            train_anno_dir), f'{train_anno_dir} is not a directory'

    random.seed(_RANDOM_SEED)

    train_dataset = None
    val_dataset = None

    if val_images_dir or val_anno_dir:
        assert os.path.isdir(
            val_images_dir), f'{val_images_dir} is not a directory'

        if annotation_format == 'coco':
            assert os.path.isfile(val_anno_dir) and val_anno_dir.endswith(
                '.json'), f'{val_anno_dir} is not a json file'
        else:
            assert os.path.isdir(
                val_anno_dir), f'{val_anno_dir} is not a directory'

        if annotation_format == 'pascal':
            train_dataset = get_dataset_info_pascal(
                image_dir=train_images_dir, anno_dir=train_anno_dir)
            val_dataset = get_dataset_info_pascal(
                image_dir=val_images_dir, anno_dir=val_anno_dir)
        elif annotation_format == 'yolo':
            train_dataset = get_dataset_info_yolo(
                image_dir=train_images_dir, anno_dir=train_anno_dir)
            val_dataset = get_dataset_info_yolo(
                image_dir=val_images_dir, anno_dir=val_anno_dir)

    elif split is not None:
        assert isinstance(split, float) or isinstance(
            split, int), f'expected to be float, but got {type(split)}\
        instead'

        assert 0 <= split <= 1.0, f'expected a float between 0 and 1, but got\
        {split} instead'

        if annotation_format == 'pascal':
            train_dataset, val_dataset = get_dataset_info_pascal(
                image_dir=train_images_dir,
                anno_dir=train_anno_dir,
                split=split)

        elif annotation_format == 'yolo':
            train_dataset, val_dataset = get_dataset_info_yolo(
                image_dir=train_images_dir,
                anno_dir=train_anno_dir,
                split=split)

    else:
        if annotation_format == 'pascal':
            train_dataset = get_dataset_info_pascal(
                image_dir=train_images_dir, anno_dir=train_anno_dir)
        elif annotation_format == 'yolo':
            train_dataset = get_dataset_info_yolo(
                image_dir=train_images_dir, anno_dir=train_anno_dir)
    if val_dataset is not None:
        return train_dataset, val_dataset
    else:
        return train_dataset


class Predictor(ABC):
    """docstring for Predictor."""

    def __init__(self, model, preprocessing_func):
        self.model = model
        self.preprocessing_func = preprocessing_func

    def load_image(self, image_path):
        return np.array(keras.preprocessing.image.load_img(path=image_path))

    @abstractmethod
    def predict(self, image):
        pass


def annotate_image(image_array,
                   bboxes,
                   scores,
                   labels,
                   threshold=0.5,
                   label_dict=None):
    image = Image.fromarray(image_array)
    Imagedraw = ImageDraw.Draw(image)

    for box, label, score in zip(bboxes, labels, scores):
        if score < threshold:
            continue

        (left, top, right, bottom) = box

        label_to_display = label
        if isinstance(label_dict, dict):
            label_to_display = label_dict[label]

        caption = '{}|{:.3f}'.format(label_to_display, score)

        colortofill = STANDARD_COLORS[label]
        Imagedraw.rectangle([left, top, right, bottom],
                            fill=None,
                            outline=colortofill)

        display_str_heights = font.getsize(caption)[1]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        text_width, text_height = font.getsize(caption)
        margin = np.ceil(0.05 * text_height)
        Imagedraw.rectangle([(left, text_bottom - text_height - 2 * margin),
                             (left + text_width, text_bottom)],
                            fill=colortofill)

        Imagedraw.text((left + margin, text_bottom - text_height - margin),
                       caption,
                       fill='black',
                       font=font)

    return image


def download_aerial_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'  # noqa: E501
    path_to_zip_file = tf.keras.utils.get_file(
        'aerial-vehicles-dataset.zip',
        zip_url,
        cache_dir=dataset_path,
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path,
                                           'aerial-vehicles-dataset')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset',
                              'images')
    annotation_dir = os.path.join(dataset_path, 'aerial-vehicles-dataset',
                                  'annotations', 'pascalvoc_xml')

    return images_dir, annotation_dir


def download_chess_dataset(dataset_path=tempfile.gettempdir()):
    zip_url = 'https://public.roboflow.ai/ds/uBYkFHtqpy?key=HZljsh2sXY'
    path_to_zip_file = tf.keras.utils.get_file(
        'chess_pieces.zip',
        zip_url,
        cache_dir=dataset_path,
        cache_subdir='',
        extract=False)
    directory_to_extract_to = os.path.join(dataset_path, 'chess_pieces')
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    images_dir = os.path.join(dataset_path, 'chess_pieces', 'train')
    annotation_dir = os.path.join(dataset_path, 'chess_pieces', 'train')

    return images_dir, annotation_dir
