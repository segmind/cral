import concurrent.futures
import json
import os
from itertools import repeat

import numpy as np
import pandas as pd
import tqdm

from .cral_hash import hashFile
from .cral_util import find_images

_RANDOM_SEED = 12


def find_classes(img_path):
    classes = []
    for folder in os.listdir(img_path):
        class_images_path = os.path.join(img_path, folder)
        if os.path.isdir(class_images_path):
            images = find_images(class_images_path)
            if (len(images) != 0):
                classes.append(folder)
    return classes


def make_dict(train_images_dir, val_images_dir):
    classes = find_classes(train_images_dir)
    json_data = {}
    json_data['task_type'] = 'Classification'
    json_data['dataset_format'] = 'Classification Format'
    json_data['num_classes'] = len(classes)
    json_data['class_names'] = classes
    json_data['train_images_dir'] = train_images_dir
    json_data['val_images_dir'] = val_images_dir
    return json_data


def get_datapoint_info_helper(func_args):
    return get_datapoint_info(func_args[0], func_args[1], func_args[2])


def get_datapoint_info(image_location, folder, train_only):
    image_name = os.path.basename(image_location)
    image_hs = hashFile(image_location)
    return image_name, folder, image_hs, train_only


def get_dataset_info(file_dir, train_only=True, split=None):
    """Parses the data and makes a dictionary.

    Args:
        file_dir (str): path to folder with images
        train_only (bool, optional): True=Train ,False=Validation
        split (float, optional): float to divide training dataset into
            training and validation

    Returns:
        dict: dictionary with information filled
    """
    master_dataset_info = {
        'image_name': [],
        'annotation_name': [],
        'image_hash': [],
        'train_only': []
    }
    master_pd_dataset = pd.DataFrame.from_dict(master_dataset_info)
    np.random.seed(_RANDOM_SEED)
    num_classes = 0
    num_images = 0
    class_name = list()
    for folder in os.listdir(file_dir):
        class_images_path = os.path.join(file_dir, folder)
        if os.path.isdir(class_images_path):
            images = find_images(class_images_path)
            if len(images) > 0:
                num_classes += 1
                class_name.append(folder)
                num_images += len(images)

    if train_only:
        subset = 'training'
    else:
        subset = 'test'
    print(f'\nProcessing {subset} dataset')
    print(
        f'Found {num_images} images belonging to {num_classes} classes with labels {class_name}'  # noqa: E501
    )

    done = 0
    done_classes = 0
    for folder in os.listdir(file_dir):
        class_images_path = os.path.join(file_dir, folder)
        if os.path.isdir(class_images_path):
            images = find_images(class_images_path)
            if len(images) > 0:
                print(
                    f"\nProcessing Class {done_classes+1}/{num_classes} '{class_name[done_classes]}'"  # noqa: E501
                )
                done_classes += 1
                dataset_info = {
                    'image_name': [],
                    'annotation_name': [],
                    'image_hash': [],
                    'train_only': []
                }
                with concurrent.futures.ThreadPoolExecutor() as executer:
                    results = list(
                        tqdm.tqdm(
                            executer.map(get_datapoint_info, images,
                                         repeat(folder), repeat(train_only)),
                            total=len(images),
                            ncols=100,
                            initial=done))
                for result in results:
                    dataset_info['image_name'].append(result[0])
                    dataset_info['annotation_name'].append(result[1])
                    dataset_info['image_hash'].append(result[2])
                    dataset_info['train_only'].append(result[3])
                pd_dataset = pd.DataFrame.from_dict(dataset_info)
                if split is not None and split != 0:
                    pd_dataset = pd_dataset.loc[np.random.permutation(
                        len(pd_dataset))].reset_index(drop=True)
                    split_len = round(len(pd_dataset) * split)
                    pd_dataset.loc[pd_dataset.index < split_len,
                                   ('train_only')] = False
                master_pd_dataset = pd.concat([master_pd_dataset, pd_dataset])
    return master_pd_dataset


def make_csv(csv_dir, train_images_dir, val_images_dir=None, split=None):
    """Parses the data and makes a csv file and returns its hash.

    Args:
        csv_dir (str): path to save the CSV file created
        train_images_dir (str): path to images
        val_images_dir (str, optional): path to validation images
        split (float, optional): float to divide training dataset into
                training and validation

    Returns:
        str: Hash of the csv file created
    """
    assert os.path.isdir(csv_dir), f'{csv_dir} is not a directory'
    assert os.path.isdir(
        train_images_dir), f'{train_images_dir} is not a directory'

    if val_images_dir:
        assert os.path.isdir(
            val_images_dir), f'{val_images_dir} is not a directory'
        dataset_df = get_dataset_info(file_dir=train_images_dir)
        val_df = get_dataset_info(file_dir=val_images_dir, train_only=False)
        dataset_df = pd.concat([dataset_df, val_df])

    elif split is not None:
        assert isinstance(split, float) or isinstance(
            split, int), f'expected to be float, but got {type(split)} instead'
        assert 0 <= split <= 1.0, f'expected a float between 0 and 1, but got {split} instead'  # noqa: E501
        dataset_df = get_dataset_info(
            file_dir=train_images_dir, train_only=True, split=split)
    else:
        dataset_df = get_dataset_info(
            file_dir=train_images_dir, train_only=True)

    json_data = make_dict(train_images_dir, val_images_dir)
    json_save_path = os.path.join(csv_dir, 'dataset.json')
    with open(json_save_path, 'w') as json_file:
        json.dump(json_data, json_file)

    dataset_save_path = os.path.join(csv_dir, 'dataset.csv')
    dataset_df.to_csv(dataset_save_path, index=False)

    dataset_csv_hash = hashFile(dataset_save_path)

    return dataset_csv_hash, dataset_save_path, json_save_path
