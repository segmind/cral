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
from albumentations import (Blur, CenterCrop, Flip, MedianBlur, MotionBlur,
                            OneOf, RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomFog, RandomGamma,
                            RandomRain, RandomRotate90, RandomShadow,
                            RandomSnow, RandomSunFlare, RGBShift, Rotate)
from botocore.exceptions import ClientError

_EPOCHS = 5
_BATCH_SIZE = 4
_STEPS = None

parser = argparse.ArgumentParser()
parser.add_argument(
    '--efficientnet', help='test EfficientNet Family', action='store_false')
parser.add_argument(
    '--resnet', help='test ResNet Family', action='store_false')
parser.add_argument(
    '--densenet', help='test DenseNet Family', action='store_false')
parser.add_argument(
    '--mobilenet', help='test MobileNet Family', action='store_false')
parser.add_argument(
    '--inception', help='test Inception Family', action='store_false')
parser.add_argument(
    '--nasnet', help='test Nasnet Family', action='store_false')
parser.add_argument(
    '--xception', help='test sception Family', action='store_false')
parser.add_argument('--vgg', help='test VGG Family', action='store_false')
parser.add_argument('--all', help='test all algorithms', action='store_false')
#parser.add_argument('unittest_args', nargs='*')

parser.add_argument('unittest_args', nargs='*')
#parser.add_argument('unittest_args', nargs=argparse.REMAINDER)

args = parser.parse_args()
# TODO: Go do something with args.input and args.filename
# Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)


class Test_Classification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from cral.tracking import set_experiment
        from cral.pipeline.core import ClassificationPipe
        from cral.augmentations.engine import Classification as Classification_augmentor

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

        set_experiment('6f2c48d8-970a-4e28-a609-38088cab599a')
        cls.new_pipe = ClassificationPipe()

        cls.new_pipe.add_data(
            train_images_dir='/tmp/bikes_persons_dataset',
            val_images_dir=None,
            split=0.2)

        cls.new_pipe.lock_data()

        aug = [
            OneOf([
                RandomFog(
                    fog_coef_lower=1, fog_coef_upper=1, alpha_coef=0.05,
                    p=1.0),
                RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=10.5, p=1.0),
                RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=0.5),
                RandomSnow(),
                RandomSunFlare()
            ],
                  p=0.2)
        ]

        augmentor = Classification_augmentor(aug=aug)
        cls.new_pipe.set_aug(augmentor)

        cls.report_path = '/tmp/reports'
        if os.path.isdir(cls.report_path) is False:
            os.mkdir(cls.report_path)

    def setup(self):
        self.new_pipe = cls.new_pipe

    # @classmethod
    # def tearDownClass(cls):

    #     from cral.tracking import data

    #     zip_path=os.path.join(cls.report_path,'reports')
    #     dir_name=cls.report_path
    #     shutil.make_archive(zip_path, 'zip', dir_name)
    #     zip_path=zip_path+'.zip'

    #     s3_client = boto3.client('s3')
    #     file_name=zip_path
    #     bucket_path='s3://segmind-builds/testing-reports/'
    #     object_name='classification_reports.zip'
    #     try:
    #         (bucket, dest_path) = data.parse_s3_uri(bucket_path)
    #         s3_client.upload_file(file_name, bucket, dest_path)
    #         #response = s3_client.upload_file(file_name, bucket, object_name)
    #     except ClientError as e:
    #         print(e)

    def upload_reports(self, dir_to_upload):
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
        self.new_pipe.set_algo(
            model=model_name,
            weights='imagenet',
            fully_connected_layer=[1024, 512],
            dropout_rate=None,
            hidden_layer_activation='relu',
            final_layer_activation='softmax',
            base_trainable=False,
            preprocessing_fn=None)

        #Compile_options={}
        #Compile_options['optimizer']='adam'
        #Compile_options['loss']=tf.keras.losses.CategoricalCrossentropy()
        #Compile_options['metrics']=['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

        self.new_pipe.train(
            num_epochs=_EPOCHS,
            snapshot_prefix='prefix',
            snapshot_path='/tmp',
            snapshot_every_n=2,
            height=300,
            width=300,
            batch_size=_BATCH_SIZE,
            validate_every_n=2,
            callbacks=[],
            steps_per_epoch=_STEPS,
            compile_options=None)

    def model_predict(self):

        _bike_list = [
            os.path.join('/tmp/bikes_persons_dataset/bike', i)
            for i in os.listdir('/tmp/bikes_persons_dataset/bike')[:5]
        ]
        _person_list = [
            os.path.join('/tmp/bikes_persons_dataset/person', i)
            for i in os.listdir('/tmp/bikes_persons_dataset/person')[:5]
        ]

        for label, confidence in self.new_pipe.predict(img_list=_bike_list):
            self.assertEqual(label, 0)

        for label, confidence in self.new_pipe.predict(img_list=_person_list):
            self.assertEqual(label, 1)

    def make_report(self, name):

        data = self.new_pipe.model.history.history

        dataframe = pd.DataFrame.from_dict(data)
        #print(dataframe.head())

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
        dataset = []
        classes = []
        img_dir = '/tmp/bikes_persons_dataset'
        for folder in os.listdir(img_dir):
            classes.append(folder)
            for file in glob.glob(os.path.join(img_dir, folder, '*.jpg')):
                dataset.append((file, folder))

        classes = sorted(classes)
        name_to_index = {}
        for i, val in enumerate(classes):
            name_to_index[val] = i
        paths = [i[0] for i in dataset]
        labels = [name_to_index[i[1]] for i in dataset]

        predictions = self.new_pipe.predict(paths)
        predicted_labels = [p[0] for p in predictions]

        conf_mat = np.array(tf.math.confusion_matrix(labels, predicted_labels))

        S = ['']
        for class_name in classes:
            S.append(class_name)
        writer.writerow(S)
        for i, row in enumerate(conf_mat):
            S = [classes[i]]
            S.extend(row)
            writer.writerow(S)

    @unittest.skipIf(args.densenet and args.all, 'not specified for testing')
    def test_DenseNet121(self):
        from cral.models.classification import DenseNet121
        base_model, preprocessing_func = DenseNet121(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet121')
        # self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('DenseNet121')

    @unittest.skipIf(args.densenet and args.all, 'not specified for testing')
    def test_DenseNet169(self):
        from cral.models.classification import DenseNet169
        base_model, preprocessing_func = DenseNet169(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet169')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('DenseNet169')

    @unittest.skipIf(args.densenet and args.all, 'not specified for testing')
    def test_DenseNet201(self):
        from cral.models.classification import DenseNet201
        base_model, preprocessing_func = DenseNet201(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'DenseNet201')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('DenseNet201')

    @unittest.skipIf(args.inception and args.all, 'not specified for testing')
    def test_InceptionResNetV2(self):
        from cral.models.classification import InceptionResNetV2
        base_model, preprocessing_func = InceptionResNetV2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'InceptionResNetV2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('InceptionResNetV2')

    @unittest.skipIf(args.inception and args.all, 'not specified for testing')
    def test_InceptionV3(self):
        from cral.models.classification import InceptionV3
        base_model, preprocessing_func = InceptionV3(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'InceptionV3')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('InceptionV3')

    @unittest.skipIf(args.mobilenet and args.all, 'not specified for testing')
    def test_MobileNet(self):
        from cral.models.classification import MobileNet
        base_model, preprocessing_func = MobileNet(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'MobileNet')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('MobileNet')

    @unittest.skipIf(args.mobilenet and args.all, 'not specified for testing')
    def test_MobileNetV2(self):
        from cral.models.classification import MobileNetV2
        base_model, preprocessing_func = MobileNetV2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'MobileNetV2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('MobileNetV2')

    @unittest.skipIf(args.nasnet and args.all, 'not specified for testing')
    def test_NASNetLarge(self):
        from cral.models.classification import NASNetLarge
        base_model, preprocessing_func = NASNetLarge(
            weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        self.model_train(base_model, preprocessing_func, 'NASNetLarge')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('NASNetLarge')

    @unittest.skipIf(args.nasnet and args.all, 'not specified for testing')
    def test_NASNetMobile(self):
        from cral.models.classification import NASNetMobile
        base_model, preprocessing_func = NASNetMobile(
            weights='imagenet', include_top=False, input_shape=(300, 300, 3))
        self.model_train(base_model, preprocessing_func, 'NASNetMobile')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('NASNetMobile')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet50(self):
        from cral.models.classification import ResNet50
        base_model, preprocessing_func = ResNet50(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet50')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet50')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet101(self):
        from cral.models.classification import ResNet101
        base_model, preprocessing_func = ResNet101(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet101')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet101')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet152(self):
        from cral.models.classification import ResNet152
        base_model, preprocessing_func = ResNet152(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet152')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet152')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet50V2(self):
        from cral.models.classification import ResNet50V2
        base_model, preprocessing_func = ResNet50V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet50V2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet50V2')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet101V2(self):
        from cral.models.classification import ResNet101V2
        base_model, preprocessing_func = ResNet101V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet101V2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet101V2')

    @unittest.skipIf(args.resnet and args.all, 'not specified for testing')
    def test_ResNet152V2(self):
        from cral.models.classification import ResNet152V2
        base_model, preprocessing_func = ResNet152V2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'ResNet152V2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('ResNet152V2')

    @unittest.skipIf(args.vgg and args.all, 'not specified for testing')
    def test_VGG16(self):
        from cral.models.classification import VGG16
        base_model, preprocessing_func = VGG16(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'VGG16')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('VGG16')

    @unittest.skipIf(args.vgg and args.all, 'not specified for testing')
    def test_VGG19(self):
        from cral.models.classification import VGG19
        base_model, preprocessing_func = VGG19(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'VGG19')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('VGG19')

    @unittest.skipIf(args.xception and args.all, 'not specified for testing')
    def test_Xception(self):
        from cral.models.classification import Xception
        base_model, preprocessing_func = Xception(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'Xception')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('Xception')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB0(self):
        from cral.models.classification import EfficientNetB0
        base_model, preprocessing_func = EfficientNetB0(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB0')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB0')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB1(self):
        from cral.models.classification import EfficientNetB1
        base_model, preprocessing_func = EfficientNetB1(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB1')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB1')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB2(self):
        from cral.models.classification import EfficientNetB2
        base_model, preprocessing_func = EfficientNetB2(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB2')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB2')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB3(self):
        from cral.models.classification import EfficientNetB3
        base_model, preprocessing_func = EfficientNetB3(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB3')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB3')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB4(self):
        from cral.models.classification import EfficientNetB4
        base_model, preprocessing_func = EfficientNetB4(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB4')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB4')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB5(self):
        from cral.models.classification import EfficientNetB5
        base_model, preprocessing_func = EfficientNetB5(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB5')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB5')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB6(self):
        from cral.models.classification import EfficientNetB6
        base_model, preprocessing_func = EfficientNetB6(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB6')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB6')

    @unittest.skipIf(args.efficientnet and args.all,
                     'not specified for testing')
    def test_EfficientNetB7(self):
        from cral.models.classification import EfficientNetB7
        base_model, preprocessing_func = EfficientNetB7(
            weights='imagenet', include_top=False)
        self.model_train(base_model, preprocessing_func, 'EfficientNetB7')
        #self.model_predict()
        tf.keras.backend.clear_session()
        self.make_report('EfficientNetB7')


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
