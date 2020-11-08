import json
import os
import tempfile

import numpy as np
import pandas as pd
import tqdm

from .cral_hash import hashFile
from .cral_util import fileName, find_images

_ALLOWED_ANNOTATION_FORMATS = ('grayscale', 'rgb')
_RANDOM_SEED = 12


def find_classes_coco(anno_path):
    classes = []
    with open(anno_path) as file:
        data = json.load(file)
        for class_desc in data['categories']:
            classes.append(class_desc['name'])
    return classes


def find_classes(anno_path, annotation_format):
    classes = []
    if annotation_format == 'coco':
        classes = find_classes_coco(anno_path)
    return classes


def make_dict(anno_path, annotation_format):
    # classes=find_classes(anno_path,annotation_format)
    json_data = {}
    # json_data["task_type"]="Segmentation"
    # json_data["dataset_format"]=annotation_format
    # json_data["num_classes"]=len(classes)
    # json_data["class_names"]=classes
    return json_data


def get_dataset_info(image_dir,
                     anno_dir,
                     annotation_format,
                     img_to_anno,
                     train_only=True):
    """Parses the data and makes a dictionary.

    Args:
        image_dir (str): path to image directory
        anno_dir (str): path to annotation directory or json file in coco
        annotation_format (str): one of "yolo","coco","pascal"
        img_to_anno (function): Function to convert image name to annotation
                name
        train_only (bool, optional): True=Train ,False=Validation

    Returns:
        dict: dictionary with information filled
    """
    print('Versioning data ...')
    dataset_info = {
        'image_name': [],
        'annotation_name': [],
        'image_hash': [],
        'annotation_hash': [],
        'train_only': []
    }
    # if annotation_format == "coco" :

    #     with open(anno_dir) as anno_file:
    #         json_file=json.load(anno_file)
    #         anno_hs=hashFile(anno_dir)
    #         anno_name=fileName(anno_dir,ext=True)
    #         for image_data in tqdm.tqdm(json_file["images"]) :
    #             image_name=image_data["file_name"]
    #             image_path=os.path.join(image_dir,image_name)
    #             if os.path.isfile(image_path):
    #                 dataset_info["image_name"].append(image_name)
    #                 dataset_info["annotation_name"].append(anno_name)
    #                 dataset_info["image_hash"].append(hashFile(image_path))
    #                 dataset_info["annotation_hash"].append(anno_hs)
    #                 dataset_info["train_only"].append(train_only)

    # else:

    images = find_images(image_dir)
    for curr_image_path in tqdm.tqdm(images):
        # curr_image_name = fileName(curr_image_path)
        if img_to_anno is not None:
            curr_anno_name = img_to_anno(fileName(curr_image_path))
        else:
            curr_anno_name = fileName(curr_image_path)
        curr_anno_path = os.path.join(anno_dir, curr_anno_name + '.png')

        assert os.path.isfile(curr_anno_path), 'cannot locate :: {}'.format(
            curr_anno_path)
        # if os.path.isfile(curr_anno_path):
        dataset_info['image_name'].append(fileName(curr_image_path, ext=True))
        dataset_info['annotation_name'].append(
            fileName(curr_anno_path, ext=True))
        dataset_info['image_hash'].append(hashFile(curr_image_path))
        dataset_info['annotation_hash'].append(hashFile(curr_anno_path))
        dataset_info['train_only'].append(train_only)

    return dataset_info


def make_csv(annotation_format,
             train_images_dir,
             train_anno_dir,
             val_images_dir=None,
             val_anno_dir=None,
             split=0.2,
             csv_dir=tempfile.gettempdir(),
             img_to_anno=(lambda a: a)):
    """Parses the data and makes a csv file and returns its hash.

    Args:
        csv_dir (str): path to save the CSV file created
        annotation_format (str): one of "coco" or "pascal"
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
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
    assert annotation_format in _ALLOWED_ANNOTATION_FORMATS,\
        f'supported annotation formats are {_ALLOWED_ANNOTATION_FORMATS}'
    # assert isinstance(img_to_anno,types.FunctionType),
    # f"img_to_ano expects a function type but recived {type(img_to_anno)}"
    # if annotation_format=="coco":
    #     assert os.path.isfile(train_anno_dir) and train_anno_dir.endswith(".json"), f"{train_anno_dir} is not a json file"  # noqa: E501
    # else:
    assert os.path.isdir(
        train_anno_dir), f'{train_anno_dir} is not a directory'

    if val_images_dir or val_anno_dir:

        assert os.path.isdir(
            val_images_dir), f'{val_images_dir} is not a directory'

        # if annotation_format=="coco":
        #     assert os.path.isfile(val_anno_dir) and train_anno_dir.endswith(".json"), f"{val_anno_dir} is not a json file"  # noqa: E501
        # else:
        assert os.path.isdir(
            val_anno_dir), f'{val_anno_dir} is not a directory'

        train_dataset_info = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            annotation_format=annotation_format,
            img_to_anno=img_to_anno)
        dataset_df = pd.DataFrame.from_dict(train_dataset_info)
        val_dataset_info = get_dataset_info(
            image_dir=val_images_dir,
            anno_dir=val_anno_dir,
            annotation_format=annotation_format,
            img_to_anno=img_to_anno,
            train_only=False)
        val_df = pd.DataFrame.from_dict(val_dataset_info)
        dataset_df = pd.concat([dataset_df, val_df])

    elif split is not None:
        assert isinstance(split, float) or isinstance(
            split, int), f'expected to be float, but got {type(split)} instead'
        assert 0 <= split <= 1.0, f'expected a float between 0 and 1, but got {split} instead'  # noqa: E501
        train_dataset_info = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            annotation_format=annotation_format,
            img_to_anno=img_to_anno)
        dataset_df = pd.DataFrame.from_dict(train_dataset_info)
        if split > 0:
            np.random.seed(_RANDOM_SEED)
            dataset_df = dataset_df.loc[np.random.permutation(
                len(dataset_df))].reset_index(drop=True)
            split_len = round(len(dataset_df) * split)
            dataset_df.loc[dataset_df.index < split_len,
                           ('train_only')] = False
    else:
        train_dataset_info = get_dataset_info(
            image_dir=train_images_dir,
            anno_dir=train_anno_dir,
            annotation_format=annotation_format,
            img_to_anno=img_to_anno)
        dataset_df = pd.DataFrame.from_dict(train_dataset_info)

    # json_data=make_dict(train_anno_dir,annotation_format)
    json_save_path = os.path.join(csv_dir, 'dataset.json')
    # with open(json_save_path,"w") as json_file:
    #     json.dump(json_data,json_file)

    dataset_save_path = os.path.join(csv_dir, 'dataset.csv')
    dataset_df.to_csv(dataset_save_path, index=False)

    dataset_csv_hash = hashFile(dataset_save_path)

    return dataset_csv_hash, dataset_save_path, json_save_path
