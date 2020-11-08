import json
import os
import tempfile

from cral.metrics.object_detection.utils import (check_gt_catIds, coco_res,
                                                 voc_to_coco_gt)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_ALLOWED_ANNOTATION_FORMATS = ('coco', 'pascal_voc')

dataset_json = os.path.join(tempfile.gettempdir(), 'dataset.json')
with open(dataset_json, 'r') as f:
    dataset_dict = json.load(f)


def coco_mAP(prediction_func=None,
             test_images_dir=None,
             test_anno_dir=None,
             annotation_format='pascal_voc'):

    assert os.path.isdir(
        test_images_dir), f'{test_images_dir} is not a directory'
    assert isinstance(
        annotation_format, str
    ), f'annotation_format has to be of type str but got {type(annotation_format)} instead'  # noqa: E501
    annotation_format = annotation_format.lower()
    assert annotation_format in _ALLOWED_ANNOTATION_FORMATS, f'supported annotation formats are {_ALLOWED_ANNOTATION_FORMATS}'  # noqa: E501
    if annotation_format == 'coco':
        assert os.path.isfile(test_anno_dir) and test_anno_dir.endswith(
            '.json'), f'{test_anno_dir} is not a json file'
    else:
        assert os.path.isdir(
            test_anno_dir), f'{test_anno_dir} is not a directory'

    # Generate json file for ground truth
    if annotation_format == 'pascal_voc':
        path_coco_gt = voc_to_coco_gt(
            image_dir=test_images_dir,
            annotation_dir=test_anno_dir,
            label_list=dataset_dict['classes'])
        fix_gt_catIds = False

    elif annotation_format == 'coco':
        path_coco_gt = test_anno_dir
        fix_gt_catIds, num_categories = check_gt_catIds(
            path_coco_gt, dataset_dict['num_classes']
        )  # return True when gt json has extra categories
        if fix_gt_catIds is True:
            print(
                f"expected {dataset_dict['num_classes']} categories but got {num_categories} instead in {os.path.split(path_coco_gt)[-1]}"  # noqa: E501
            )

    # Generate json file for the predictions
    path_coco_res = coco_res(
        path_coco_gt=path_coco_gt,
        prediction_func=prediction_func,
        image_dir=test_images_dir,
        score_threshold=0.5,
        annotation_format=annotation_format,
        autofix=fix_gt_catIds)

    # initiate COCO API
    coco_gt = COCO(path_coco_gt)
    coco_dt = coco_gt.loadRes(path_coco_res)
    iouType = 'bbox'

    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType=iouType)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
