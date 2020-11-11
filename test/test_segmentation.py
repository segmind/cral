import glob
import os
import tempfile
import unittest
import zipfile

import cv2
import numpy as np
import tensorflow as tf


class Test_SegmentationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/segmentation/mini_ADE20K.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'mini_ADE20K.zip',
            zip_url,
            cache_dir=tempfile.gettempdir(),
            cache_subdir='',
            extract=False)
        directory_to_extract_to = os.path.join(tempfile.gettempdir(),
                                               'mini_ADE20K')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.dataset = os.path.join(directory_to_extract_to, 'mini_ADE20K')

    def setup(self):
        self.dataset = cls.dataset

    def test_UNet(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import UNetConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='mobilenet', config=UNetConfig(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_unet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_fpnNet(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import FpnNetConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='mobilenet', config=FpnNetConfig(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_fpnet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_pspNet(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import PspNetConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(feature_extractor='mobilenet', config=PspNetConfig())

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_pspnet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_segNet(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import SegNetConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='mobilenet', config=SegNetConfig(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_segnet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_UNetPlusPlus(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import UnetPlusPlusConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='mobilenet', config=UnetPlusPlusConfig(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_unetplusplus',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_deeplabv3(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import Deeplabv3Config

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='resnet50', config=Deeplabv3Config(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_deeplabv3',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()

    def test_linkNet(self):
        from cral.pipeline import SemanticSegPipe
        from cral.models.semantic_segmentation import LinkNetConfig

        pipe = SemanticSegPipe()

        pipe.add_data(
            train_images_dir=os.path.join(self.dataset, 'images'),
            train_anno_dir=os.path.join(self.dataset, 'annotations'),
            annotation_format='rgb',
            split=0.2)

        pipe.lock_data()

        pipe.set_algo(
            feature_extractor='mobilenet', config=LinkNetConfig(224, 224))

        pipe.train(
            num_epochs=2,
            snapshot_prefix='test_linknet',
            snapshot_path='/tmp',
            snapshot_every_n=10,
            batch_size=1,
            steps_per_epoch=2)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
