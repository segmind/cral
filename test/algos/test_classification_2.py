import argparse
import csv
import glob
import os
import pathlib
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
from botocore.exceptions import ClientError

_EPOCHS = 10
_BATCH_SIZE = 2
_STEPS = None

parser = argparse.ArgumentParser()
parser.add_argument(
    '--efficientnet', help='test EfficientNet Family', action='store_true')
parser.add_argument('--resnet', help='test ResNet Family', action='store_true')
parser.add_argument(
    '--densenet', help='test DenseNet Family', action='store_true')
parser.add_argument(
    '--mobilenet', help='test MobileNet Family', action='store_true')
parser.add_argument(
    '--inception', help='test Inception Family', action='store_true')
parser.add_argument('--nasnet', help='test Nasnet Family', action='store_true')
parser.add_argument(
    '--xception', help='test xception Family', action='store_true')
parser.add_argument(
    '--darknet', help='test darknet Family', action='store_true')
parser.add_argument('--vgg', help='test VGG Family', action='store_true')
parser.add_argument('--all', help='test all algorithms', action='store_true')

parser.add_argument(
    '--efficientnetb0', help='test EfficientNetB0', action='store_true')
parser.add_argument(
    '--efficientnetb1', help='test EfficientNetB1', action='store_true')
parser.add_argument(
    '--efficientnetb2', help='test EfficientNetB2', action='store_true')
parser.add_argument(
    '--efficientnetb3', help='test EfficientNetB3', action='store_true')
parser.add_argument(
    '--efficientnetb4', help='test EfficientNetB4', action='store_true')
parser.add_argument(
    '--efficientnetb5', help='test EfficientNetB5', action='store_true')
parser.add_argument(
    '--efficientnetb6', help='test EfficientNetB6', action='store_true')
parser.add_argument(
    '--efficientnetb7', help='test EfficientNetB7', action='store_true')

parser.add_argument('--resnet50', help='test ResNet50', action='store_true')
parser.add_argument('--resnet101', help='test ResNet101', action='store_true')
parser.add_argument('--resnet152', help='test ResNet152', action='store_true')
parser.add_argument(
    '--resnet50v2', help='test ResNet50v2', action='store_true')
parser.add_argument(
    '--resnet101v2', help='test ResNet101v2', action='store_true')
parser.add_argument(
    '--resnet152v2', help='test ResNet152v2', action='store_true')

parser.add_argument(
    '--densenet121', help='test DenseNet121', action='store_true')
parser.add_argument(
    '--densenet169', help='test DenseNet169', action='store_true')
parser.add_argument(
    '--densenet201', help='test DenseNet201', action='store_true')

parser.add_argument(
    '--mobilenetv1', help='test MobileNetV1', action='store_true')
parser.add_argument(
    '--mobilenetv2', help='test MobileNetV2', action='store_true')

parser.add_argument(
    '--inceptionresnetv2', help='test InceptionResNetV2', action='store_true')
parser.add_argument(
    '--inceptionv3', help='test InceptionV3', action='store_true')

parser.add_argument(
    '--nasnetlarge', help='test NASNetLarge', action='store_true')
parser.add_argument(
    '--nasnetmobile', help='test NASNetMobile', action='store_true')

parser.add_argument('--vgg16', help='test VGG16', action='store_true')
parser.add_argument('--vgg19', help='test VGG19', action='store_true')

parser.add_argument('-e', '--epochs', help='Set Epochs for training', type=int)
parser.add_argument(
    '-b', '--batch_size', help='Set batch_size for training', type=int)
parser.add_argument(
    '-s', '--steps', help='Set steps_per_epoch for training', type=int)

#parser.add_argument('unittest_args', nargs='*')

parser.add_argument('unittest_args', nargs='*')
#parser.add_argument('unittest_args', nargs=argparse.REMAINDER)

args = parser.parse_args()
# TODO: Go do something with args.input and args.filename
# Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)

skip_DenseNet121 = not (args.all or args.densenet121 or args.densenet)
skip_DenseNet169 = not (args.all or args.densenet169 or args.densenet)
skip_DenseNet201 = not (args.all or args.densenet201 or args.densenet)

skip_InceptionResNetV2 = not (args.all or args.inceptionresnetv2
                              or args.inception)
skip_InceptionV3 = not (args.all or args.inceptionv3 or args.inception)

skip_MobileNet = not (args.all or args.mobilenetv1 or args.mobilenet)
skip_MobileNetV2 = not (args.all or args.mobilenetv2 or args.mobilenet)

skip_NASNetLarge = not (args.all or args.nasnetlarge or args.nasnet)
skip_NASNetMobile = not (args.all or args.nasnetmobile or args.nasnet)

skip_ResNet50 = not (args.all or args.resnet50 or args.resnet)
skip_ResNet101 = not (args.all or args.resnet101 or args.resnet)
skip_ResNet152 = not (args.all or args.resnet152 or args.resnet)
skip_ResNet50V2 = not (args.all or args.resnet50v2 or args.resnet)
skip_ResNet101V2 = not (args.all or args.resnet101v2 or args.resnet)
skip_ResNet152V2 = not (args.all or args.resnet152v2 or args.resnet)

skip_VGG16 = not (args.all or args.vgg16 or args.vgg)
skip_VGG19 = not (args.all or args.vgg19 or args.vgg)

skip_EfficientNetB0 = not (args.all or args.efficientnetb0
                           or args.efficientnet)
skip_EfficientNetB1 = not (args.all or args.efficientnetb1
                           or args.efficientnet)
skip_EfficientNetB2 = not (args.all or args.efficientnetb2
                           or args.efficientnet)
skip_EfficientNetB3 = not (args.all or args.efficientnetb3
                           or args.efficientnet)
skip_EfficientNetB4 = not (args.all or args.efficientnetb4
                           or args.efficientnet)
skip_EfficientNetB5 = not (args.all or args.efficientnetb5
                           or args.efficientnet)
skip_EfficientNetB6 = not (args.all or args.efficientnetb6
                           or args.efficientnet)
skip_EfficientNetB7 = not (args.all or args.efficientnetb7
                           or args.efficientnet)

skip_Xception = not (args.all or False or args.xception)
skip_darknet53 = not (args.all or False or args.darknet)

if (args.epochs is not None):
    _EPOCHS = args.epochs

if (args.batch_size is not None):
    _BATCH_SIZE = args.batch_size

if (args.steps is not None):
    _STEPS = args.steps


class Test_Classification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from cral.tracking import set_experiment
        from cral.pipeline.core import ClassificationPipe
        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/classification/bikes_persons_dataset.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'bikes_persons_dataset.zip',
            zip_url,
            cache_dir='/tmp',
            cache_subdir='',
            extract=False)
        directory_to_extract_to = '/tmp'
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.data_dir = '/tmp/bikes_persons_dataset'

        cls.data_dir = pathlib.Path(cls.data_dir)
        image_count = len(list(cls.data_dir.glob('*/*.jpg')))
        cls.CLASS_NAMES = sorted(
            np.array([
                item.name for item in cls.data_dir.glob('*')
                if item.name != 'LICENSE.txt'
            ]))
        cls.IMG_HEIGHT = 300
        cls.IMG_WIDTH = 300
        cls._STEPS = image_count / _BATCH_SIZE
        cls.report_path = '/tmp/reports'
        if os.path.isdir(cls.report_path) is False:
            os.mkdir(cls.report_path)

    def setup(self):

        self.data_dir = cls.data_dir
        self.CLASS_NAMES = cls.CLASS_NAMES
        self.IMG_HEIGHT = cls.IMG_HEIGHT
        self.IMG_WIDTH = cls.IMG_WIDTH
        self.report_path = cls.report_path
        self.images = cls.images
        self.labels = cls.labels
        self._STEPS = cls._STEPS

    def upload_reports(self, dir_to_upload):
        return 0
        from cral.tracking import data

        zip_path = dir_to_upload
        dir_name = dir_to_upload
        shutil.make_archive(zip_path, 'zip', dir_name)
        zip_path = zip_path + '.zip'

        s3_client = boto3.client('s3')
        file_name = zip_path
        bucket_path = 's3://segmind-builds/testing-reports/'
        #object_name='classification_reports.zip'
        try:
            (bucket, dest_path) = data.parse_s3_uri(bucket_path)
            dest_path = posixpath.join(dest_path, os.path.basename(zip_path))
            s3_client.upload_file(file_name, bucket, dest_path)
            #response = s3_client.upload_file(file_name, bucket, object_name)
        except ClientError as e:
            print(e)

    def model_train(self, base_model, preprocessing_func, model_name):

        from cral.models.classification import densely_connected_head
        from tensorflow.keras.layers import Dense
        from tensorflow.keras import Model

        print('\n' + model_name + '\n')
        for layer in base_model.layers:
            layer.trainable = False

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            preprocessing_function=preprocessing_func,
            validation_split=0.2)

        self.train_data_gen = image_generator.flow_from_directory(
            directory=str(self.data_dir),
            batch_size=_BATCH_SIZE,
            shuffle=True,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            classes=list(self.CLASS_NAMES),
            subset='training')

        self.validation_data_gen = image_generator.flow_from_directory(
            directory=str(self.data_dir),
            batch_size=_BATCH_SIZE,
            shuffle=True,
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            classes=list(self.CLASS_NAMES),
            subset='validation')

        self.images = []
        for img_path in self.train_data_gen.filepaths:
            self.images.append(
                np.array(
                    tf.keras.preprocessing.image.load_img(
                        path=img_path,
                        target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))))
        self.labels = self.train_data_gen.labels
        self.images = preprocessing_func(np.array(self.images))

        num_classes = len(self.CLASS_NAMES)

        final_hidden_layer = densely_connected_head(
            base_model, [1024, 512],
            dropout_rate=None,
            hidden_layer_Activation='relu')

        output = Dense(
            units=num_classes, activation='softmax')(
                final_hidden_layer)

        self.model = Model(
            inputs=base_model.inputs,
            outputs=output,
            name='{}_custom'.format(model_name))

        # self.new_pipe.set_algo(model,preprocessing_func)

        Compile_options = {}
        Compile_options['optimizer'] = tf.keras.optimizers.Adam(lr=1e-4)
        Compile_options['loss'] = tf.keras.losses.CategoricalCrossentropy()
        Compile_options['metrics'] = [
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
        self._STEP = None
        if (_STEPS is not None and _STEPS < self._STEPS):
            self._STEPS = _STEPS
        self.model.compile(**Compile_options)
        self.model.fit(
            self.train_data_gen,
            epochs=_EPOCHS,
            batch_size=_BATCH_SIZE,
            steps_per_epoch=self._STEP,
            validation_data=self.validation_data_gen)

    def model_predict(self):
        return 0
        _bike_list = [
            os.path.join('/tmp/bikes_persons_dataset/bike', i)
            for i in os.listdir('/tmp/bikes_persons_dataset/bike')[:5]
        ]
        _person_list = [
            os.path.join('/tmp/bikes_persons_dataset/person', i)
            for i in os.listdir('/tmp/bikes_persons_dataset/person')[:5]
        ]

        for label, confidence in self.model.predict(self.train_data_gen):
            self.assertEqual(label, 0)

        for label, confidence in self.model.predict(self.train_data_gen):
            self.assertEqual(label, 1)

    def make_report(self, name):

        data = self.model.history.history

        dataframe = pd.DataFrame.from_dict(data)
        #print(.head())

        save_dir = os.path.join(self.report_path, name)

        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        algo_metrics_csv = os.path.join(save_dir,
                                        '{}_metrics.csv'.format(name))
        dataframe.to_csv(algo_metrics_csv, index_label='S.No')

        confusion_matrix_path = os.path.join(
            save_dir, '{}_confusion_matrix.csv'.format(name))
        with open(confusion_matrix_path, 'w') as file:
            csv_writer = csv.writer(file)
            self.confusion_matrix(csv_writer)

        self.upload_reports(save_dir)

    def confusion_matrix(self, writer):
        predictions = self.model.predict(self.images)
        predicted_labels = [np.argmax(p) for p in predictions]
        # print(predicted_labels)
        conf_mat = np.array(
            tf.math.confusion_matrix(self.labels, predicted_labels))

        S = ['']
        for class_name in self.CLASS_NAMES:
            S.append(class_name)
        writer.writerow(S)
        for i, row in enumerate(conf_mat):
            S = [self.CLASS_NAMES[i]]
            S.extend(row)
            writer.writerow(S)

    @unittest.skipIf(skip_DenseNet121, 'not specified for testing')
    def test_DenseNet121(self):
        from cral.models.classification import DenseNet121
        base_model, preprocessing_func = DenseNet121(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet121')
        # self.model_predict()
        self.make_report('DenseNet121')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_DenseNet169, 'not specified for testing')
    def test_DenseNet169(self):
        from cral.models.classification import DenseNet169
        base_model, preprocessing_func = DenseNet169(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet169')
        # self.model_predict()
        self.make_report('DenseNet169')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_DenseNet201, 'not specified for testing')
    def test_DenseNet201(self):
        from cral.models.classification import DenseNet201
        base_model, preprocessing_func = DenseNet201(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet201')
        # self.model_predict()
        self.make_report('DenseNet201')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB0, 'not specified for testing')
    def test_EfficientNetB0(self):
        from cral.models.classification import EfficientNetB0
        base_model, preprocessing_func = EfficientNetB0(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB0')
        # self.model_predict()
        self.make_report('EfficientNetB0')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB1, 'not specified for testing')
    def test_EfficientNetB1(self):
        from cral.models.classification import EfficientNetB1
        base_model, preprocessing_func = EfficientNetB1(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB1')
        # self.model_predict()
        self.make_report('EfficientNetB1')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB2, 'not specified for testing')
    def test_EfficientNetB2(self):
        from cral.models.classification import EfficientNetB2
        base_model, preprocessing_func = EfficientNetB2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB2')
        # self.model_predict()
        self.make_report('EfficientNetB2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB3, 'not specified for testing')
    def test_EfficientNetB3(self):
        from cral.models.classification import EfficientNetB3
        base_model, preprocessing_func = EfficientNetB3(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB3')
        # self.model_predict()
        self.make_report('EfficientNetB3')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB4, 'not specified for testing')
    def test_EfficientNetB4(self):
        from cral.models.classification import EfficientNetB4
        base_model, preprocessing_func = EfficientNetB4(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB4')
        # self.model_predict()
        self.make_report('EfficientNetB4')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB5, 'not specified for testing')
    def test_EfficientNetB5(self):
        from cral.models.classification import EfficientNetB5
        base_model, preprocessing_func = EfficientNetB5(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB5')
        # self.model_predict()
        self.make_report('EfficientNetB5')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB6, 'not specified for testing')
    def test_EfficientNetB6(self):
        from cral.models.classification import EfficientNetB6
        base_model, preprocessing_func = EfficientNetB6(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB6')
        # self.model_predict()
        self.make_report('EfficientNetB6')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_EfficientNetB7, 'not specified for testing')
    def test_EfficientNetB7(self):
        from cral.models.classification import EfficientNetB7
        base_model, preprocessing_func = EfficientNetB7(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB7')
        # self.model_predict()
        self.make_report('EfficientNetB7')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_InceptionResNetV2, 'not specified for testing')
    def test_InceptionResNetV2(self):
        from cral.models.classification import InceptionResNetV2
        base_model, preprocessing_func = InceptionResNetV2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'InceptionResNetV2')
        # self.model_predict()
        self.make_report('InceptionResNetV2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_InceptionV3, 'not specified for testing')
    def test_InceptionV3(self):
        from cral.models.classification import InceptionV3
        base_model, preprocessing_func = InceptionV3(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'InceptionV3')
        # self.model_predict()
        self.make_report('InceptionV3')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_MobileNet, 'not specified for testing')
    def test_MobileNet(self):
        from cral.models.classification import MobileNet
        base_model, preprocessing_func = MobileNet(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'MobileNet')
        # self.model_predict()
        self.make_report('MobileNet')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_MobileNetV2, 'not specified for testing')
    def test_MobileNetV2(self):
        from cral.models.classification import MobileNetV2
        base_model, preprocessing_func = MobileNetV2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'MobileNetV2')
        # self.model_predict()
        self.make_report('MobileNetV2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_NASNetLarge, 'not specified for testing')
    def test_NASNetLarge(self):
        from cral.models.classification import NASNetLarge
        base_model, preprocessing_func = NASNetLarge(
            weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        self.model_train(base_model, preprocessing_func, 'NASNetLarge')
        # self.model_predict()
        self.make_report('NASNetLarge')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_NASNetMobile, 'not specified for testing')
    def test_NASNetMobile(self):
        from cral.models.classification import NASNetMobile
        base_model, preprocessing_func = NASNetMobile(
            weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        self.model_train(base_model, preprocessing_func, 'NASNetMobile')
        # self.model_predict()
        self.make_report('NASNetMobile')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet101, 'not specified for testing')
    def test_ResNet101(self):
        from cral.models.classification import ResNet101
        base_model, preprocessing_func = ResNet101(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet101')
        # self.model_predict()
        self.make_report('ResNet101')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet101V2, 'not specified for testing')
    def test_ResNet101V2(self):
        from cral.models.classification import ResNet101V2
        base_model, preprocessing_func = ResNet101V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet101V2')
        # self.model_predict()
        self.make_report('ResNet101V2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet152, 'not specified for testing')
    def test_ResNet152(self):
        from cral.models.classification import ResNet152
        base_model, preprocessing_func = ResNet152(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet152')
        # self.model_predict()
        self.make_report('ResNet152')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet152V2, 'not specified for testing')
    def test_ResNet152V2(self):
        from cral.models.classification import ResNet152V2
        base_model, preprocessing_func = ResNet152V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet152V2')
        # self.model_predict()
        self.make_report('ResNet152V2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet50, 'not specified for testing')
    def test_ResNet50(self):
        from cral.models.classification import ResNet50
        base_model, preprocessing_func = ResNet50(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet50')
        # self.model_predict()
        self.make_report('ResNet50')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_ResNet50V2, 'not specified for testing')
    def test_ResNet50V2(self):
        from cral.models.classification import ResNet50V2
        base_model, preprocessing_func = ResNet50V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet50V2')
        # self.model_predict()
        self.make_report('ResNet50V2')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_VGG16, 'not specified for testing')
    def test_VGG16(self):
        from cral.models.classification import VGG16
        base_model, preprocessing_func = VGG16(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'VGG16')
        # self.model_predict()
        self.make_report('VGG16')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_VGG19, 'not specified for testing')
    def test_VGG19(self):
        from cral.models.classification import VGG19
        base_model, preprocessing_func = VGG19(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'VGG19')
        # self.model_predict()
        self.make_report('VGG19')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_Xception, 'not specified for testing')
    def test_Xception(self):
        from cral.models.classification import Xception
        base_model, preprocessing_func = Xception(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'Xception')
        # self.model_predict()
        self.make_report('Xception')
        tf.keras.backend.clear_session()

    @unittest.skipIf(skip_darknet53, 'not specified for testing')
    def test_darknet53(self):
        from cral.models.classification import Darknet53
        base_model, preprocessing_func = Darknet53(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'darknet53')
        # self.model_predict()
        self.make_report('darknet53')
        tf.keras.backend.clear_session()


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('efficientnet', help="test EfficientNet Family", action="store_true")
    # parser.add_argument('resnet', help="test ResNet Family", action="store_true")
    # parser.add_argument('densenet', help="test DenseNet Family", action="store_true")
    # parser.add_argument('mobilenet', help="test MobileNet Family", action="store_true")
    # parser.add_argument('inception', help="test Inception Family", action="store_true")
    # parser.add_argument('nasnet', help="test Nasnet Family", action="store_true")
    # parser.add_argument('xception', help="test sception Family", action="store_true")
    # parser.add_argument('vgg', help="test VGG Family", action="store_true")
    # parser.add_argument('unittest_args', nargs='*')

    # args = parser.parse_args()
    # # TODO: Go do something with args.input and args.filename
    # test_xception_flag = args.xception
    # test_efficientnet_flag = args.efficientnet
    # test_mobilenet_flag = args.mobilenet
    # test_inception_flag = args.inception
    # test_nasnet_flag = args.nasnet
    # test_vgg_flag = args.vgg
    # test_densenet_flag = args.densenet
    # test_resnet_flag = args.resnet
    # # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
    sys.argv[1:] = args.unittest_args
    unittest.main()
