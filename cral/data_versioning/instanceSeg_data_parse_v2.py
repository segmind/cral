import pandas as pd
import os
import tqdm
import json
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from cral.data_versioning.cral_hash import hashFile

_ALLOWED_ANNOTATION_FORMATS = ("coco")
_EXT = {"pascal_voc": ".xml", "yolo": ".txt"}
_PARALLEL_READS = 16
_RANDOM_SEED = 12

_ALLOWED_ANNOTATION_FORMATS = ("coco")
_EXT = {"pascal_voc": ".xml", "yolo": ".txt"}
_PARALLEL_READS = 16
_RANDOM_SEED = 12


def find_classes(dataset):
    return sorted(dataset['label_name'].unique().tolist())


def set_up_label_ids(dataset):
    _CLASSES = find_classes(dataset)
    _CLASSES.insert(0, 'BG')
    label_to_id = {}
#     if "background_empty" in _CLASSES:
#         label_to_id["background_empty"] = -1
#         _CLASSES.remove("background_empty")
    for i, name in enumerate(_CLASSES):
        label_to_id[name] = i
#     print(label_to_id)
    dataset['label_id'] = dataset['label_name'].map(label_to_id)
    return dataset


def give_dataset_dict(dataset, train_images_dir, train_anno_dir,
                      val_images_dir, val_anno_dir):
    _CLASSES = find_classes(dataset)
    _CLASSES.insert(0, 'BG')
#     if "background_empty" in _CLASSES:
#         _CLASSES.remove("background_empty")
#     _CLASSES.append('background')
#     _CLASSES.insert(0, 'BG')
    meta_info = {'task_type': 'instance_segmentation',
                 'train_image_dir': train_images_dir,
                 'train_anno_dir': train_anno_dir,
                 'val_image_dir': val_images_dir,
                 'val_anno_dir': val_anno_dir,
                 'num_classes': len(_CLASSES),
                 'classes': _CLASSES}
    return meta_info


def coco_image_info(image_info):
    image_info['image_hash'] = hashFile(image_info['image_path'])


def give_coco_datapoint(image, anno, dataset_info, cat_id_to_cat):
    xmin = anno['bbox'][0]
    ymin = anno['bbox'][1]
    xmax = anno['bbox'][0]+anno['bbox'][2]
    ymax = anno['bbox'][1]+anno['bbox'][3]
    segmentation = anno['segmentation']
#     print(segmentation)
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
    dataset_info['segmentation'].append(segmentation)
    dataset_info['label_name'].append(label_name)


def get_dataset_info_coco(image_dir, anno_dir, start_id=0, split=None,
                          train_only=True, *args, **kwargs):
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
    print(f"\nFinding {subset} images")
    for image in tqdm.tqdm(anno_file['images'], ncols=100):
        image_name = image['file_name']
        image_path = os.path.join(image_dir, image_name)
        if os.path.isfile(image_path):
            image_info = {'image_id': 0,
                          'image_name': image_name,
                          'image_hash': None,
                          'image_height': image['height'],
                          'image_width': image['width'],
                          'orignal_id': str(image['id']),
                          'image_path': image_path}
            image_info_list.append(image_info)
            image_num += 1
    print(f"\nProcessing {subset} Images")
    with ThreadPoolExecutor() as executer:
        results = list(tqdm.tqdm(executer.map(
            coco_image_info, image_info_list), total=len(image_info_list),
            ncols=100))

    for image in image_info_list:
        id_to_image[str(image['orignal_id'])] = image

    dataset_info = {'image_id': [],
                    'image_name': [],
                    'image_hash': [],
                    'image_width': [],
                    'image_height': [],
                    'xmin': [],
                    'ymin': [],
                    'xmax': [],
                    'ymax': [],
                    'segmentation': [],
                    'label_name': [],
                    'train_only': []}

    print(f"\nProcessing {subset} Annotations")
    cat_id_to_cat = {}
    for cat in anno_file['categories']:
        cat_id_to_cat[str(cat['id'])] = cat['name']

    for anno in tqdm.tqdm(anno_file['annotations'], ncols=100):

        orignal_id = str(anno['image_id'])

        if orignal_id in id_to_image:
            image = id_to_image[orignal_id]
            give_coco_datapoint(image, anno, dataset_info, cat_id_to_cat)

    dataset_info['train_only'] = repeat(
        train_only, len(dataset_info['image_name']))
    dataset_info['image_id'] = repeat(0, len(dataset_info['image_name']))
    dataset_df = pd.DataFrame.from_dict(dataset_info)
    grouped = [df for _, df in dataset_df.groupby('image_name')]
    random.shuffle(grouped)
    total_images = len(grouped)
    dataset_df = pd.concat(grouped).reset_index(drop=True)
    dataset_df['image_id'] = dataset_df.groupby(
        'image_name', sort=False).ngroup()+start_id
    if split != 0 and split is not None:
        split_len = split*total_images
        dataset_df.loc[dataset_df['image_id'] <
                       start_id+split_len, 'train_only'] = False

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
    """Parses the data and makes a csv file and returns its hash
    Args:
        annotation_format (str): one of "yolo","coco","pascal_voc"
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        csv_dir (str): path to save the CSV file created
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        names_file (None, optional): Path to .names file in YOLO format
        split (float, optional): float to divide training dataset into
        traing and val
        img_to_anno (function, optional): Function to convert image name
        to annotation name
    Returns:
        str: Hash of the csv file created
    """

    assert os.path.isdir(csv_dir), f"{csv_dir} is not a directory"
    assert os.path.isdir(
      train_images_dir), f"{train_images_dir} is not a directory"
    assert isinstance(annotation_format,
                      str), f"annotation_format has to be of type str but got \
    {type(annotation_format)} instead"
    annotation_format = annotation_format.lower()
    assert annotation_format in _ALLOWED_ANNOTATION_FORMATS, f"supported\
    annotation formats are {_ALLOWED_ANNOTATION_FORMATS}"

    if annotation_format == "coco":
        assert os.path.isfile(train_anno_dir) and train_anno_dir.endswith(
          ".json"), f"{train_anno_dir} is not a json file"
    else:
        assert os.path.isdir(
          train_anno_dir), f"{train_anno_dir} is not a directory"

    get_dataset_info = None
    random.seed(_RANDOM_SEED)

    if annotation_format == 'coco':
        get_dataset_info = get_dataset_info_coco
    # elif annotation_format == 'yolo':
    #     get_dataset_info = get_dataset_info_yolo
    # elif annotation_format == 'pascal_voc':
    #     get_dataset_info = get_dataset_info_pascal

    if val_images_dir or val_anno_dir:
        assert os.path.isdir(
          val_images_dir), f"{val_images_dir} is not a directory"
        if annotation_format == "coco":
            assert os.path.isfile(val_anno_dir) and val_anno_dir.endswith(
              ".json"), f"{val_anno_dir} is not a json file"
        else:
            assert os.path.isdir(
              val_anno_dir), f"{val_anno_dir} is not a directory"

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
          split, int), f"expected to be float, but got {type(split)} instead"
        assert 0 <= split <= 1.0, f"expected a float between 0 and 1, but got\
        {split} instead"
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
    meta_into = give_dataset_dict(dataset_df,
                                  train_images_dir,
                                  train_anno_dir,
                                  val_images_dir,
                                  val_anno_dir)
    with open(json_save_path, "w") as json_file:
        json.dump(meta_into, json_file)

    return hashFile(csv_save_path), csv_save_path, json_save_path
