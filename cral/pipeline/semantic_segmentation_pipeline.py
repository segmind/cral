import json
import os

import jsonpickle
import tensorflow as tf
# from cral.tracking import get_experiment, log_artifact, KerasCallback, log_param  # noqa: E501
# from cral.callbacks import checkpoint_callback
from cral.data_feeder.semantic_seg_data_feeder import \
    create_tfrecords as create_tfrecords_semantic_segmentation
from cral.data_versioning import segmentation_dataset_hasher
from cral.models.semantic_segmentation import SparseMeanIoU
from cral.models.semantic_segmentation.deeplabv3 import Deeplabv3Config
from cral.models.semantic_segmentation.FpnNet import FpnNetConfig
from cral.models.semantic_segmentation.LinkNet import LinkNetConfig
from cral.models.semantic_segmentation.PspNet import PspNetConfig
from cral.models.semantic_segmentation.SegNet import SegNetConfig
from cral.models.semantic_segmentation.Unet import UNetConfig
from cral.models.semantic_segmentation.UnetPlusPlus import UnetPlusPlusConfig
from cral.pipeline.core import PipelineBase
from tensorflow import keras


class SemanticSegPipe(PipelineBase):
    """docstring for SemanticSegPipe."""

    def __init__(self, *args, **kwargs):
        super(SemanticSegPipe, self).__init__(
            task_type='Semantic_Segmentation', *args, **kwargs)
        self.dcrf = True

    def add_data(self,
                 train_images_dir,
                 train_anno_dir,
                 annotation_format,
                 val_images_dir=None,
                 val_anno_dir=None,
                 names_file=None,
                 split=None,
                 img_to_anno=None):
        """Parses dataset once for generating metadata and versions the data.

        Args:
            train_images_dir (str): path to images
            train_anno_dir (str): path to annotation
            annotation_format (str): one of "yolo","coco","pascal"
            val_images_dir (str, optional): path to validation images
            val_anno_dir (str, optional): path to vallidation annotation
            names_file (None, optional): Path to .names file in YOLO format
            split (float, optional): float to divide training dataset into
                training and val
            img_to_anno (function, optional): Function to convert image name
                to annotation name
        """
        self.dataset_hash, self.dataset_csv_path, self.dataset_json = segmentation_dataset_hasher(  # noqa: E501
            annotation_format=annotation_format,
            train_images_dir=train_images_dir,
            train_anno_dir=train_anno_dir,
            val_images_dir=val_images_dir,
            val_anno_dir=val_anno_dir,
            # names_file=names_file,
            split=split,
            img_to_anno=img_to_anno)

        # No need to save because it is empty
        # with open(self.dataset_json) as f:
        #     self.data_dict = json.loads(f.read())
        if split is not None:
            val_images_dir = train_images_dir
            val_anno_dir = train_anno_dir
        self.data_dict = dict(
            train_images_dir=train_images_dir,
            train_anno_dir=train_anno_dir,
            val_images_dir=val_images_dir,
            val_anno_dir=val_anno_dir)

        # rewrite because the above returns only path where to write json
        with open(self.dataset_json, 'w') as json_file:
            json_file.write(json.dumps(self.data_dict, indent=2))

        # print(self.data_dict)
        self.update_project_file(self.data_dict)
        self.update_project_file({'annotation_format': annotation_format})

    def lock_data(self):
        """Parse Data and makes tf-records and creates meta-data."""
        meta_info = create_tfrecords_semantic_segmentation(
            self.data_dict, self.dataset_csv_path)

        self.update_project_file(meta_info)

    def set_aug(self):
        pass

    def set_algo(self,
                 feature_extractor,
                 config,
                 weights='imagenet',
                 base_trainable=False,
                 preprocessing_fn=None,
                 optimizer=keras.optimizers.Adam(lr=1e-4, clipnorm=0.001),
                 distribute_strategy=None):

        num_classes = int(self.cral_meta_data['num_classes'])
        architecture = None

        self.preprocessing_fn = None

        feature_extractor = feature_extractor.lower()

        if isinstance(config, Deeplabv3Config):
            from cral.models.semantic_segmentation import (
                create_DeepLabv3Plus, log_deeplabv3_config_params)

            assert isinstance(
                config,
                Deeplabv3Config), 'please provide a `Deeplabv3Config()` object'

            if feature_extractor == 'mobilenetv2':  # hardcoded for now
                config.height = 224
                config.width = 224
                config.input_shape = (224, 224, 3)

            log_deeplabv3_config_params(config)

            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_DeepLabv3Plus(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_DeepLabv3Plus(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'deeplabv3plus'

        elif isinstance(config, UNetConfig):
            from cral.models.semantic_segmentation import (
                create_UNet, log_UNet_config_params)

            assert isinstance(
                config, UNetConfig), 'please provide a `UNetConfig()` object'

            log_UNet_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_UNet(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_UNet(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'UNet'

        elif isinstance(config, PspNetConfig):
            print('entered pspnet')
            from cral.models.semantic_segmentation import (
                create_PspNet, log_PspNet_config_params)

            assert isinstance(
                config,
                PspNetConfig), 'please provide a `PspNetConfig()` object'

            log_PspNet_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_PspNet(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_PspNet(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'PspNet'

        elif isinstance(config, SegNetConfig):
            print('entered segnet')
            from cral.models.semantic_segmentation import (
                create_SegNet, log_SegNet_config_params)

            assert isinstance(
                config,
                SegNetConfig), 'please provide a `SegNetConfig()` object'

            log_SegNet_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_SegNet(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_SegNet(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))

            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'SegNet'

        elif isinstance(config, FpnNetConfig):
            print('entered fpnNet')
            from cral.models.semantic_segmentation import (
                create_FpnNet, log_FpnNet_config_params)

            assert isinstance(
                config,
                FpnNetConfig), 'please provide a `FpnNetConfig()` object'

            log_FpnNet_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_FpnNet(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_FpnNet(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'FpnNet'

        elif isinstance(config, UnetPlusPlusConfig):
            print('entered unet++')
            from cral.models.semantic_segmentation import (
                create_UnetPlusPlus, log_UnetPlusPlus_config_params)

            assert isinstance(
                config, UnetPlusPlusConfig
            ), 'please provide a `UnetPlusPlusConfig()` object'

            log_UnetPlusPlus_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_UnetPlusPlus(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_UnetPlusPlus(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'UnetPlusPlus'

        elif isinstance(config, LinkNetConfig):
            print('entered LinkNet')
            from cral.models.semantic_segmentation import \
                create_LinkNet, log_LinkNet_config_params

            assert isinstance(
                config,
                LinkNetConfig), 'please provide a `LinkNetConfig()` object'

            log_LinkNet_config_params(config)
            base_trainable = True
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_LinkNet(
                    feature_extractor, config, num_classes, weights,
                    base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'semanic_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['semanic_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'semanic_segmentation_meta'].keys():
                                old_extractor = dic[
                                    'semanic_segmentation_meta'][
                                        'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_LinkNet(
                    feature_extractor, config, num_classes, None,
                    base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # deeplabv3 default losses
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
            architecture = 'LinkNet'

        else:
            raise ValueError('argument to `config` is not understood.')

        # custom image normalizing function
        if preprocessing_fn is not None:
            self.preprocessing_fn = preprocessing_fn

        @tf.function(
            input_signature=[tf.TensorSpec([None, None, 3], dtype=tf.float32)])
        def _preprocess(image_array):
            """tf.function-deocrated version of preprocess_"""
            im_arr = self.preprocessing_fn(image_array)
            input_batch = tf.expand_dims(im_arr, axis=0)
            return input_batch

        # Attach function to Model
        self.model.preprocess = _preprocess

        # Model parallelism
        if distribute_strategy is None:
            self.model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=['accuracy',
                         SparseMeanIoU(num_classes=num_classes)])
        else:
            with distribute_strategy.scope():
                self.model.compile(
                    loss=loss,
                    optimizer=optimizer,
                    metrics=[
                        'accuracy',
                        SparseMeanIoU(num_classes=num_classes)
                    ])

        algo_meta = dict(
            feature_extractor=feature_extractor,
            architecture=architecture,
            weights=weights,
            base_trainable=base_trainable,
            config=jsonpickle.encode(config))

        semanic_segmentation_meta = dict(semanic_segmentation_meta=algo_meta)

        self.update_project_file(semanic_segmentation_meta)

    def visualize_data(self):
        pass

    def train(self,
              num_epochs,
              snapshot_prefix,
              snapshot_path,
              snapshot_every_n,
              batch_size=2,
              validation_batch_size=None,
              validate_every_n=1,
              callbacks=[],
              steps_per_epoch=None,
              compile_options=None,
              distribute_strategy=None,
              log_evry_n_step=100):

        assert isinstance(num_epochs,
                          int), 'num epochs to run should be in `int`'
        assert isinstance(callbacks, list)
        assert isinstance(validate_every_n, int)

        snapshot_prefix = str(snapshot_prefix)

        if validation_batch_size is None:
            validation_batch_size = batch_size

        # num_classes = int(self.cral_meta_data['num_classes'])
        training_set_size = int(self.cral_meta_data['num_training_images'])
        test_set_size = int(self.cral_meta_data['num_test_images'])

        if self.model is None:
            raise ValueError(
                'please define a model first using set_algo() function')

        if compile_options is not None:
            assert isinstance(compile_options, dict)

            # Model parallelism
            if distribute_strategy is None:
                self.model.compile(**compile_options)
            else:
                with distribute_strategy.scope():
                    self.model.compile(**compile_options)

        meta_info = dict(
            snapshot_prefix=snapshot_prefix,
            num_epochs=num_epochs,
            batch_size=batch_size)

        self.update_project_file(meta_info)

        tfrecord_dir = self.cral_meta_data['tfrecord_path']

        train_tfrecords = os.path.join(tfrecord_dir, 'train*.tfrecord')
        test_tfrecords = os.path.join(tfrecord_dir, 'test*.tfrecord')

        if self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'deeplabv3plus':

            from cral.models.semantic_segmentation import DeepLabv3Generator

            deeplabv3_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                deeplabv3_config, Deeplabv3Config
            ), 'Expected an instance of cral.models.semantic_segmentation.Deeplabv3Config'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = DeepLabv3Generator(
                config=deeplabv3_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'UNet':

            from cral.models.semantic_segmentation import UNetGenerator

            unet_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                unet_config, UNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.UNetConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = UNetGenerator(
                config=unet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'PspNet':

            from cral.models.semantic_segmentation import PspNetGenerator

            pspnet_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                pspnet_config, PspNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.PspNetConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = PspNetGenerator(
                config=pspnet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'FpnNet':

            from cral.models.semantic_segmentation import FpnNetGenerator

            fpnNet_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                fpnNet_config, FpnNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.FpnNetConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = FpnNetGenerator(
                config=fpnNet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'SegNet':

            from cral.models.semantic_segmentation import SegNetGenerator

            segnet_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                segnet_config, SegNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.SegNetConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = SegNetGenerator(
                config=segnet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'UnetPlusPlus':

            from cral.models.semantic_segmentation import UnetPlusPlusGenerator

            unetplusplus_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                unetplusplus_config, UnetPlusPlusConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.UnetPlusPlusConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = UnetPlusPlusGenerator(
                config=unetplusplus_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['semanic_segmentation_meta'][
                'architecture'] == 'LinkNet':

            from cral.models.semantic_segmentation import LinkNetGenerator

            linknet_config = jsonpickle.decode(
                self.cral_meta_data['semanic_segmentation_meta']['config'])

            assert isinstance(
                linknet_config, LinkNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.LinkNetConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = LinkNetGenerator(
                config=linknet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size // validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        else:
            raise ValueError('argument to `config` is not understood.')

        if steps_per_epoch is None:
            steps_per_epoch = training_set_size // batch_size

        # callbacks.append(KerasCallback(log_evry_n_step))
        # callbacks.append(KerasCallback())
        # callbacks.append(
        #     checkpoint_callback(
        #         snapshot_every_epoch=snapshot_every_n,
        #         snapshot_path=snapshot_path,
        #         checkpoint_prefix=snapshot_prefix,
        #         save_h5=False))

        # Attach segmind.cral as an asset
        tf.io.gfile.copy(self.cral_file, 'segmind.cral', overwrite=True)
        cral_asset_file = tf.saved_model.Asset('segmind.cral')

        self.model.cral_file = cral_asset_file

        self.model.fit(
            x=train_input_function,
            epochs=num_epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_input_function,
            validation_steps=validation_steps,
            validation_freq=validate_every_n)

        final_model_path = os.path.join(snapshot_path,
                                        str(snapshot_prefix) + '_final')

        self.model.save(filepath=final_model_path, overwrite=True)

        print('Saved the final Model to :\n {}'.format(final_model_path))

    def prediction_model(self, checkpoint_file):

        self.model = keras.models.load_model(
            checkpoint_file,
            compile=False,
            custom_objects={'SparseMeanIoU': SparseMeanIoU})

        try:
            location_to_cral_file = self.model.cral_file.asset_path.numpy()
            with open(location_to_cral_file) as f:
                metainfo = json.loads(f.read())
            # pprint(metainfo)

        except AttributeError:
            print(
                "Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted"  # noqa: E501
            )

        architecture = metainfo['semanic_segmentation_meta']['architecture']
        num_classes = int(metainfo['num_classes'])
        feature_extractor = metainfo['semanic_segmentation_meta'][
            'feature_extractor']

        if architecture == 'deeplabv3plus':
            if feature_extractor not in [
                    'xception', 'resnet50', 'mobilenetv2'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import Deeplabv3Predictor
            from cral.models.semantic_segmentation import create_DeepLabv3Plus

            deeplabv3_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                deeplabv3_config, Deeplabv3Config
            ), 'Expected an instance of cral.models.semantic_segmentation.Deeplabv3Config'  # noqa: E501

            unused_model, preprocessing_fn = create_DeepLabv3Plus(
                feature_extractor, deeplabv3_config, num_classes, weights=None)
            del (unused_model)

            pred_object = Deeplabv3Predictor(
                height=deeplabv3_config.height,
                width=deeplabv3_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict

        elif architecture == 'FpnNet':
            if feature_extractor not in [
                    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
                    'resnet101v2', 'resnet152v2', 'vgg16', 'vgg19',
                    'mobilenet', 'mobilenetv2'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import FpnNetPredictor
            from cral.models.semantic_segmentation import create_FpnNet

            fpnNet_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                fpnNet_config, FpnNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.FpnNetConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_FpnNet(
                feature_extractor, fpnNet_config, num_classes, weights=None)
            del (unused_model)

            pred_object = FpnNetPredictor(
                height=fpnNet_config.height,
                width=fpnNet_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict

        elif architecture == 'UNet':
            if feature_extractor not in [
                    'mobilenet', 'resnet50', 'resnet101', 'resnet152', 'vgg16',
                    'vgg19'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import UNetPredictor
            from cral.models.semantic_segmentation import create_UNet

            unet_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                unet_config, UNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.UNetConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_UNet(
                feature_extractor, unet_config, num_classes, weights=None)
            del (unused_model)

            pred_object = UNetPredictor(
                height=unet_config.height,
                width=unet_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict
        elif architecture == 'PspNet':
            if feature_extractor not in [
                    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
                    'resnet101v2', 'resnet152v2', 'vgg16', 'vgg19',
                    'mobilenet', 'mobilenetv2'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import PspNetPredictor
            from cral.models.semantic_segmentation import create_PspNet

            pspnet_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                pspnet_config, PspNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.PspNetConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_PspNet(
                feature_extractor, pspnet_config, num_classes, weights=None)
            del (unused_model)

            pred_object = PspNetPredictor(
                height=pspnet_config.height,
                width=pspnet_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict

        elif architecture == 'SegNet':
            if feature_extractor not in [
                    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
                    'resnet101v2', 'resnet152v2', 'vgg16', 'vgg19',
                    'mobilenet', 'mobilenetv2'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import SegNetPredictor
            from cral.models.semantic_segmentation import create_SegNet

            segnet_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                segnet_config, SegNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.SegNetConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_SegNet(
                feature_extractor, segnet_config, num_classes, weights=None)
            del (unused_model)

            pred_object = SegNetPredictor(
                height=segnet_config.height,
                width=segnet_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict

        elif architecture == 'UnetPlusPlus':
            if feature_extractor not in [
                    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
                    'resnet101v2', 'resnet152v2', 'vgg16', 'vgg19',
                    'mobilenet', 'mobilenetv2'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import UnetPlusPlusPredictor
            from cral.models.semantic_segmentation import create_UnetPlusPlus

            unetplusplus_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                unetplusplus_config, UnetPlusPlusConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.UnetPlusPlusConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_UnetPlusPlus(
                feature_extractor,
                unetplusplus_config,
                num_classes,
                weights=None)
            del (unused_model)

            pred_object = UnetPlusPlusPredictor(
                height=unetplusplus_config.height,
                width=unetplusplus_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict

        elif architecture == 'LinkNet':
            if feature_extractor not in [
                    'resnet50', 'resnet101', 'resnet152', 'resnet50v2',
                    'resnet101v2', 'resnet152v2', 'mobilenet'
            ]:
                raise ValueError(f'{feature_extractor} not yet supported ..')

            from cral.models.semantic_segmentation import LinkNetPredictor
            from cral.models.semantic_segmentation import create_LinkNet

            linknet_config = jsonpickle.decode(
                metainfo['semanic_segmentation_meta']['config'])
            assert isinstance(
                linknet_config, LinkNetConfig
            ), 'Expected an instance of cral.models.semantic_segmentation.LinkNetConfig'  # noqa: E501

            unused_model, preprocessing_fn = create_LinkNet(
                feature_extractor, linknet_config, num_classes, weights=None)
            del (unused_model)

            pred_object = LinkNetPredictor(
                height=linknet_config.height,
                width=linknet_config.width,
                model=self.model,
                preprocessing_func=preprocessing_fn,
                dcrf=self.dcrf)

            return pred_object.predict
