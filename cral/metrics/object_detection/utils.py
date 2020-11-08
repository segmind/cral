import datetime
import glob
import json
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm

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


def get_coco_annotation_from_obj(obj, label2id, image_id, annotation_id):
    label = obj.findtext('name')
    assert label in label2id, f'Error: {label} is not in label2id !'
    category_id = label2id[label]
    bndbox = obj.find('bndbox')

    xmin = int(bndbox.findtext('xmin'))
    ymin = int(bndbox.findtext('ymin'))
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f'Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}'  # noqa: E501
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'id': annotation_id,
        'image_id': image_id,
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        # 'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def create_image_info(image_id,
                      file_name,
                      image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1,
                      coco_url='',
                      flickr_url=''):

    image_info = {
        'id': image_id,
        'file_name': file_name,
        'width': image_size[0],
        'height': image_size[1],
        'date_captured': date_captured,
        'license': license_id,
        'coco_url': coco_url,
        'flickr_url': flickr_url
    }

    return image_info


def voc_to_coco_gt(image_dir, annotation_dir, label_list):

    CATEGORIES = []
    coco_output = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': CATEGORIES,
        'images': [],
        'annotations': []
    }

    label_list.sort()
    label2id = dict(zip(label_list, list(range(len(label_list)))))

    annotation_id = 1

    for image_id, xml_file in enumerate(
            tqdm(glob.glob(os.path.join(annotation_dir, '*.xml'))), 1):

        ann_tree = ET.parse(xml_file)
        ann_root = ann_tree.getroot()

        image_filepath = xml_file.replace(annotation_dir,
                                          image_dir).replace('.xml', '.jpg')
        image = Image.open(image_filepath)

        image_info = create_image_info(
            image_id=image_id,
            file_name=os.path.basename(image_filepath),
            image_size=image.size,
            coco_url=image_filepath)

        coco_output['images'].append(image_info)

        for obj in ann_root.findall('object'):
            coco_annotation = get_coco_annotation_from_obj(
                obj=obj,
                label2id=label2id,
                image_id=image_id,
                annotation_id=annotation_id)

            annotation_id += 1

            coco_output['annotations'].append(coco_annotation)

    for labelname, labelid in label2id.items():
        CATEGORIES.append({
            'id': labelid,
            'name': labelname,
            'supercategory': 'custom'
        })

    coco_output['categories'] = CATEGORIES
    coco_output = json.dumps(coco_output, indent=4)

    path_coco_gt = os.path.join(tempfile.gettempdir(), 'voctococo-gt.json')
    with open(path_coco_gt, 'w') as f:
        f.write(coco_output)

    return path_coco_gt


def check_gt_catIds(gt_json, num_classes):
    with open(gt_json, 'r') as f:
        distros_dict = json.load(f)
        num_categories = len(distros_dict['categories'])
    return num_categories != num_classes, num_categories


def fix_catIds(gt_dict, path_coco_gt):
    supercategories, categories = [], []
    cats = gt_dict['categories']
    annots = gt_dict['annotations']
    for c in cats:
        categories.append(c['name'])
        supercategories.append(c['supercategory'])

    rm_catIds = [i for i in categories if i in supercategories]
    rm_id = []
    for catId in cats:
        if catId['name'] in rm_catIds:
            rm_id.append(catId['id'])

    for i in range(len(rm_id)):
        cats.pop(rm_id[i])

    for n in range(len(cats)):
        cat = cats[n]
        for annot in annots:
            if annot['category_id'] == cat['id']:
                annot['category_id'] = n
        cat['id'] = n

    gt_dict['categories'] = cats
    gt_output = json.dumps(gt_dict, indent=4)
    with open(path_coco_gt, 'w') as f:
        f.write(gt_output)

    return gt_dict


def coco_res(path_coco_gt,
             prediction_func,
             image_dir,
             annotation_format,
             autofix,
             score_threshold=0.5):

    result_array = []

    # Get GT file
    with open(path_coco_gt, 'r') as f:
        distros_dict = json.load(f)
        if autofix:
            assert 'roboflow' in distros_dict['info']['description'].lower(
            ), f'{os.path.split(path_coco_gt)[-1]} is not in coco format'
            print('autofixing...')
            distros_dict = fix_catIds(distros_dict, path_coco_gt)

    try:
        element_0 = distros_dict['images'][0]
        image_path = element_0.get('coco_url', None)
        if image_path and os.path.isfile(image_path):
            use_coco_url = True
        else:
            use_coco_url = False
    except KeyError:
        use_coco_url = False

    for element in distros_dict['images']:
        image_id = element['id']
        if use_coco_url:
            image_path = element['coco_url']
        else:
            image_path = os.path.join(image_dir, element['file_name'])

        boxes, scores, labels = prediction_func(image_path)
        # print(boxes.shape, scores.shape, labels.shape)
        # print(type(boxes), type(scores), type(labels))
        boxes = np.array(boxes)
        boxes = boxes.astype(int)
        # bb convert x1,y1,x2,x2 (prediction output) -> xywh (coco format)
        for n in range(len(boxes)):
            bbox = boxes[n]
            bbox[2] = bbox[2] - bbox[0]  # convert to width
            bbox[3] = bbox[3] - bbox[1]  # convert to height

        for bbox, score, label in zip(boxes, scores, labels):
            if score < score_threshold:
                break
            # print(image_id, bbox, score.item(), label.item())
            results_info = {
                'image_id': image_id,
                'category_id': label.item(),
                'bbox': bbox.tolist(),
                'score': score.item()
            }
            result_array.append(results_info)

    result_array = json.dumps(result_array)
    path_coco_res = os.path.join(tempfile.gettempdir(), 'voctococo-res.json')
    with open(path_coco_res, 'w') as f:
        f.write(result_array)

    return path_coco_res
