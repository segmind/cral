import tempfile

from .classification_data_parse_v2 import \
    make_csv as classification_dataset_hasher
from .ObjectDetection_parse_data_v2 import \
    make_csv as objectDetection_dataset_hasher
from .segmentation_data_parse_v2 import make_csv as segmentation_dataset_hasher
from .instanceSeg_data_parse_v2 import make_csv as instanceSeg_dataset_hasher

def log_classification_dataset(*args, **kwargs):
    """Parses the classification data and logs to tracking server.

    Args:
        train_images_dir (str): path to images
        val_images_dir (str, optional): path to validation images
        split (float, optional): float to divide training dataset into training and validation
    """

    dataset_hash, dataset_csv_path, dataset_json = classification_dataset_hasher(
        tempfile.gettempdir(), *args, **kwargs)


def log_segmentation_dataset(*args, **kwargs):
    """Parses the segmentation data and logs to tracking server.

    Args:
        annotation_format (str): one of 'coco' or 'pascal'
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        img_to_anno (function, optional): Function to convert image name to annotation name
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        split (float, optional): float to divide training dataset into training and val
    """
    dataset_hash, dataset_csv_path, dataset_json = segmentation_dataset_hasher(
        tempfile.gettempdir(), *args, **kwargs)


def log_object_detection_dataset(*args, **kwargs):
    """Parses the object detection data and logs to tracking server.

    Args:
        annotation_format (str): one of 'yolo','coco','pascal'
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        img_to_anno (function, optional): Function to convert image name to annotation name
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        split (float, optional): float to divide training dataset into training and val
    """
    dataset_hash, dataset_csv_path, dataset_json = objectDetection_dataset_hasher(
        tempfile.gettempdir(), *args, **kwargs)

def log_instance_segmentation_dataset(*args, **kwargs):
    """Parses the instance segmentation data and logs to tracking server

    Args:
        annotation_format (str): one of 'coco'
        train_images_dir (str): path to images
        train_anno_dir (str): path to annotation
        img_to_anno (function, optional): Function to convert image name to annotation name
        val_images_dir (str, optional): path to validation images
        val_anno_dir (str, optional): path to vallidation annotation
        split (float, optional): float to divide training dataset into training and val
    """
    dataset_hash, dataset_csv_path, dataset_json = instanceSeg_dataset_hasher(
        tempfile.gettempdir(), *args, **kwargs)
    try_mlflow_log(log_artifact, key='dataset_versioned.csv', path=dataset_csv_path)
    try_mlflow_log(log_artifact, key='dataset_meta.json', path=dataset_json)
