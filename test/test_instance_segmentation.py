import glob
import os
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import tensorflow as tf


class Test_InstanceSegmentationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        zip_url = 'http://images.cocodataset.org/zips/val2017.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'val2017.zip',
            zip_url,
            cache_dir=tempfile.gettempdir(),
            cache_subdir='',
            extract=False)
        directory_to_extract_to = os.path.join(tempfile.gettempdir(),
                                               'coco2017')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.dataset = os.path.join(directory_to_extract_to, 'val2017')

        zip_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'  # noqa: E501
        path_to_zip_file = tf.keras.utils.get_file(
            'annotations_trainval2017.zip',
            zip_anno_url,
            cache_dir=tempfile.gettempdir(),
            cache_subdir='',
            extract=False)
        directory_to_extract_to = os.path.join(tempfile.gettempdir(),
                                               'coco2017_annotations')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.annotations = os.path.join(directory_to_extract_to,
                                       'annotations')

    def setup(self):
        self.dataset = cls.dataset
        self.annotations = cls.annotations

    def test_MaskRCNN(self):
        from cral.pipeline import InstanceSegPipe
        from cral.models.instance_segmentation import MaskRCNNConfig

        pipe = InstanceSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset),
            train_anno_dir=os.path.join(self.annotations,
                                        'instances_val2017.json'),
            annotation_format='coco',
            split=0.2)

        meta_info = pipe.lock_data()

        pipe.set_algo(
            feature_extractor='resnet101',
            config=MaskRCNNConfig(height=256,
                                  width=256),
            weights='imagenet')

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_mrcnn',
            snapshot_path='/tmp',
            snapshot_every_n=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
