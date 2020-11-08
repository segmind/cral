import argparse
import csv
import glob
import os
import posixpath
import random
import shutil
import sys
import time
import unittest
import zipfile

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class Test_Retinanet(unittest.TestCase):
    """docstring for Test_Retinanet."""

    # def setup(self):
    #     from cral.models.object_detection.retinanet.base import get_retinanet

    def test_retinanet_r50(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet50

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet50(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_r50v2(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet50v2

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet50v2(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_r101(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet101

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet101(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_r101v2(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet101v2

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet101v2(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_r152(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet152

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet152(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_r152v2(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_resnet152v2

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_resnet152v2(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_d121(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_densenet121

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_densenet121(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_d169(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_densenet169

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_densenet169(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_d201(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_densenet201

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_densenet201(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_mobile(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_mobilenet

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_mobilenet(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_mobilev2(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_mobilenetv2

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_mobilenetv2(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_xception(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_xception

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_xception(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_vgg16(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_vgg16

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_vgg16(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_vgg19(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_vgg19

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_vgg19(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb0(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb0

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb0(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb1(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb1

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb1(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb2(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb2

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb2(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb3(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb3

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb3(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb4(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb4

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb4(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb5(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb5

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb5(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()

    def test_retinanet_efficientnetb6(self):
        from cral.models.object_detection import RetinanetConfig, retinanet_efficientnetb6

        config = RetinanetConfig()
        model, preprocessing_fn = retinanet_efficientnetb6(
            num_classes=4,
            num_anchors_per_location=config.num_anchors(),
            weights=None)
        # print(model.summary())
        keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()
