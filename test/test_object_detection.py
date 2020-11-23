import glob
import os
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import tensorflow as tf


class Test_DetectionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'aerial-vehicles-dataset.zip',
            zip_url,
            cache_dir=tempfile.gettempdir(),
            cache_subdir='',
            extract=False)
        directory_to_extract_to = os.path.join(tempfile.gettempdir(),
                                               'aerial-vehicles-dataset')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.dataset = directory_to_extract_to

    def setup(self):
        self.dataset = cls.dataset

    def test_retinanet(self):
        from cral.pipeline import ObjectDetectionPipe
        from cral.models.object_detection import RetinanetConfig

        pipe = ObjectDetectionPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations',
                                        'pascalvoc_xml'),
            annotation_format='pascal_voc',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(feature_extractor='resnet50', config=RetinanetConfig())

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_retinanet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_yolov3(self):
        from cral.pipeline import ObjectDetectionPipe
        from cral.models.object_detection import YoloV3Config

        pipe = ObjectDetectionPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations',
                                        'pascalvoc_xml'),
            annotation_format='pascal_voc',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(feature_extractor='darknet53', config=YoloV3Config())

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_yolov3',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_ssd(self):
        from cral.pipeline import ObjectDetectionPipe
        from cral.models.object_detection import SSD300Config

        pipe = ObjectDetectionPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations',
                                        'pascalvoc_xml'),
            annotation_format='pascal_voc',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(feature_extractor='vgg16', config=SSD300Config())

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_ssd',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_fasterrcnn(self):
        from cral.pipeline import ObjectDetectionPipe
        from cral.models.object_detection import FasterRCNNConfig

        pipe = ObjectDetectionPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations',
                                        'pascalvoc_xml'),
            annotation_format='pascal_voc',
            split=0.2)

        meta_info = pipe.lock_data()

        pipe.set_algo(
            feature_extractor='resnet101',
            config=FasterRCNNConfig(height=256, width=256),
            weights='imagenet')

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_fasterrcnn',
            snapshot_path='/tmp',
            snapshot_every_n=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
