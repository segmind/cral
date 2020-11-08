import os
import time
import unittest
import zipfile
from urllib.request import urlretrieve

import tensorflow as tf


class TestClassification(unittest.TestCase):

    def setUp(self):
        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/classification/bikes_persons_dataset.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'bikes_persons_dataset.zip',
            zip_url,
            cache_dir='/tmp',
            cache_subdir='',
            extract=False)
        #path_to_zip_file = '/tmp/bikes_persons_dataset.zip'
        directory_to_extract_to = '/tmp'
        #urlretrieve(zip_url, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    def test_directory_format_with_split(self):
        from cral.data_versioning import classification_dataset_hasher

        hash_of_dataset = classification_dataset_hasher(
            csv_dir='/tmp',
            train_images_dir='/tmp/bikes_persons_dataset',
            val_images_dir=None,
            split=0.2)

    def test_directory_format_with_val(self):
        from cral.data_versioning import classification_dataset_hasher

        hash_of_dataset = classification_dataset_hasher(
            csv_dir='/tmp',
            train_images_dir='/tmp/bikes_persons_dataset',
            val_images_dir='/tmp/bikes_persons_dataset',
            split=0.0)

    def test_log_dataset(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from cral.tracking import set_experiment
        from cral.data_versioning import log_classification_dataset

        set_experiment('6f2c48d8-970a-4e28-a609-38088cab599a')

        log_classification_dataset(
            train_images_dir='/tmp/bikes_persons_dataset',
            val_images_dir='/tmp/bikes_persons_dataset',
            split=0.0)


class TestObjectDetection(unittest.TestCase):

    def setUp(self):
        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'aerial-vehicles-dataset.zip',
            zip_url,
            cache_dir='/tmp',
            cache_subdir='',
            extract=False)
        #path_to_zip_file = '/tmp/aerial-vehicles-dataset.zip'
        directory_to_extract_to = '/tmp'
        #urlretrieve(zip_url, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

    def test_directory_format_with_split(self):
        from cral.data_versioning import objectDetection_dataset_hasher

        train_img = '/tmp/images'
        anno_yolo = '/tmp/annotations/yolo_txt'
        anno_pascal = '/tmp/annotations/pascalvoc_xml'
        csv_dir = '/tmp'

        hash_of_dataset = objectDetection_dataset_hasher(
            csv_dir=csv_dir,
            annotation_format='pascal_voc',
            train_images_dir=train_img,
            train_anno_dir=anno_pascal,
            val_images_dir=None,
            val_anno_dir=None,
            split=0.2)

        hash_of_dataset = objectDetection_dataset_hasher(
            csv_dir=csv_dir,
            annotation_format='yolo',
            train_images_dir=train_img,
            train_anno_dir=anno_yolo,
            val_images_dir=None,
            val_anno_dir=None,
            split=0.2)

    def test_directory_format_with_val(self):
        from cral.data_versioning import objectDetection_dataset_hasher

        train_img = '/tmp/images'
        anno_yolo = '/tmp/annotations/yolo_txt'
        anno_pascal = '/tmp/annotations/pascalvoc_xml'
        csv_dir = '/tmp'

        hash_of_dataset = objectDetection_dataset_hasher(
            csv_dir=csv_dir,
            annotation_format='pascal_voc',
            train_images_dir=train_img,
            train_anno_dir=anno_pascal,
            val_images_dir=train_img,
            val_anno_dir=anno_pascal,
            split=None)

        hash_of_dataset = objectDetection_dataset_hasher(
            csv_dir=csv_dir,
            annotation_format='yolo',
            train_images_dir=train_img,
            train_anno_dir=anno_yolo,
            val_images_dir=train_img,
            val_anno_dir=anno_yolo,
            split=None)

    def test_log_dataset(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from cral.tracking import set_experiment
        from cral.data_versioning import log_object_detection_dataset

        set_experiment('6f2c48d8-970a-4e28-a609-38088cab599a')

        train_img = '/tmp/images'
        anno_yolo = '/tmp/annotations/yolo_txt'
        anno_pascal = '/tmp/annotations/pascalvoc_xml'
        csv_dir = '/tmp'

        log_object_detection_dataset(
            annotation_format='yolo',
            train_images_dir=train_img,
            train_anno_dir=anno_yolo,
            val_images_dir=train_img,
            val_anno_dir=anno_yolo,
            split=None)


# class TestSegmentation(unittest.TestCase):

#   def setUp(self):
#       zip_url = 'https://www.kaggle.com/sayantandas30011998/zanzibar-openai-building-footprint-mapping/download'
#       path_to_zip_file = '/tmp/zanzibar-openai-building-footprint-mapping.zip'
#       directory_to_extract_to = '/tmp'
#       urlretrieve(zip_url, path_to_zip_file)
#       with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
#           zip_ref.extractall(directory_to_extract_to)

#   def test_directory_format_with_split(self):
#       from cral.data_versioning import segmentation_dataset_hasher

#       train_img="/tmp/znz-segment-z19/znz-train-z19-all-buffered/images-512"
#       anno_pascal="/tmp/znz-segment-z19/znz-train-z19-all-buffered/masks-512"
#       csv_dir="/tmp"

#       def func(img):
#           return img[:-3]+"mask_buffered"

#       hash_of_dataset = segmentation_dataset_hasher(
#           csv_dir=csv_dir,
#           train_images_dir=train_img,
#           train_anno_dir=anno_pascal,
#           annotation_format='pascal',
#           img_to_anno=func,
#           val_images_dir=None,
#           val_anno_dir=None,
#           split=0.2)

#   def test_directory_format_with_val(self):
#       from cral.data_versioning import segmentation_dataset_hasher

#       train_img="/tmp/znz-segment-z19/znz-train-z19-all-buffered/images-512"
#       anno_pascal="/tmp/znz-segment-z19/znz-train-z19-all-buffered/masks-512"
#       csv_dir="/tmp"

#       def func(img):
#           return img[:-3]+"mask_buffered"

#       hash_of_dataset = segmentation_dataset_hasher(
#           csv_dir=csv_dir,
#           train_images_dir=train_img,
#           train_anno_dir=anno_pascal,
#           annotation_format='pascal',
#           img_to_anno=func,
#           val_images_dir=train_img,
#           val_anno_dir=anno_pascal,
#           split=None)

if __name__ == '__main__':
    unittest.main()
