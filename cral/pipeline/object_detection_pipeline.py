import tensorflow as tf
from tensorflow import keras
import os
import json
import jsonpickle
import numpy as np
import cv2
from PIL import Image
from cral.pipeline.core import PipelineBase
from cral.data_versioning import objectDetection_dataset_hasher
from cral.data_feeder.object_detection_parallel_data_feeder import \
        create_tfrecords as create_tfrecords_object_detection
from cral.common import classification_networks

# from cral.tracking import get_experiment, log_artifact, KerasCallback, log_param  # noqa: E501
from cral.models.object_detection.FasterRCNN import (FasterRCNNConfig,
                                                     log_FasterRCNN_config_params)  # noqa: E501
from cral.models.object_detection.retinanet import (RetinanetConfig,
                                                    get_retinanet_fromconfig,
                                                    log_retinanet_config_params
                                                    )
from cral.models.object_detection.SSD import (SSD300Config, decode_detections)
# from cral.callbacks import checkpoint_callback
from cral.models.object_detection.YoloV3 import (YoloV3Config,
                                                 log_yolo_config_params,
                                                 create_yolo_model)
from tensorflow.python.eager.context import eager_mode, graph_mode


class ObjectDetectionPipe(PipelineBase):
    """Cral pipeline for classification task."""

    def __init__(self, *args, **kwargs):
        super(ObjectDetectionPipe, self).__init__(
            task_type='object_Detection', *args, **kwargs)

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
            img_to_anno (function, optional): Function to convert image name to
                    annotation name
        """
        self.dataset_hash, self.dataset_csv_path, self.dataset_json = objectDetection_dataset_hasher(  # noqa: E501
            annotation_format=annotation_format,
            train_images_dir=train_images_dir,
            train_anno_dir=train_anno_dir,
            val_images_dir=val_images_dir,
            val_anno_dir=val_anno_dir,
            names_file=names_file,
            split=split,
            img_to_anno=img_to_anno)

        with open(self.dataset_json) as f:
            self.data_dict = json.loads(f.read())
        self.update_project_file(self.data_dict)
        self.update_project_file({'annotation_format': annotation_format})

    def lock_data(self):
        """Parse Data and makes tf-records and creates meta-data."""
        meta_info = create_tfrecords_object_detection(self.dataset_json,
                                                      self.dataset_csv_path)

        self.update_project_file(meta_info)
        return meta_info

    def set_aug(self, aug):
        self.aug_pipeline = aug

    def visualize_data(self):
        # tfrecord_dir = self.cral_meta_data['tfrecord_path']

        # train_tfrecords = os.path.join(tfrecord_dir, 'train*.tfrecord')
        # test_tfrecords = os.path.join(tfrecord_dir, 'test*.tfrecord')
        return

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
        if isinstance(config, YoloV3Config):
            from cral.models.object_detection.YoloV3 import Yolo_Loss
            log_yolo_config_params(config)

            self.objectdetector = 'yolo'

            assert feature_extractor.lower(
            ) == 'darknet53', 'Only DarkNet53 feature extractor is supported for Yolo'  # noqa: E501s
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = create_yolo_model(
                    config,
                    feature_extractor=feature_extractor,
                    num_classes=num_classes,
                    base_trainable=base_trainable,
                    weights=weights)

            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'object_detection_meta' in dic.keys():
                            if 'config' in dic['object_detection_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['object_detection_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'object_detection_meta'].keys():
                                old_extractor = dic['object_detection_meta'][
                                    'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                elif not (config.check_equality(old_config)):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = create_yolo_model(
                    config,
                    feature_extractor=feature_extractor,
                    num_classes=num_classes,
                    base_trainable=base_trainable,
                    weights=None)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))

            else:
                assert False, 'Weights file is not supported'

            l1_candidate_anchors = config.anchors[config.anchor_mask[0]]
            l2_candidate_anchors = config.anchors[config.anchor_mask[1]]
            l3_candidate_anchors = config.anchors[config.anchor_mask[2]]

            layer1_loss = Yolo_Loss(
                input_shape=config.input_shape,
                candidate_anchors=l1_candidate_anchors,
                grid_shape=self.model.outputs[0].shape,
                num_classes=num_classes)
            layer2_loss = Yolo_Loss(
                input_shape=config.input_shape,
                candidate_anchors=l2_candidate_anchors,
                grid_shape=self.model.outputs[1].shape,
                num_classes=num_classes)
            layer3_loss = Yolo_Loss(
                input_shape=config.input_shape,
                candidate_anchors=l3_candidate_anchors,
                grid_shape=self.model.outputs[2].shape,
                num_classes=num_classes)
            loss = {
                'tf_op_layer_y1_pred': layer1_loss,
                'tf_op_layer_y2_pred': layer2_loss,
                'tf_op_layer_y3_pred': layer3_loss
            }
            architecture = 'yolo'

        # SSD300
        elif isinstance(config, SSD300Config):
            # log_ssd_config_params(config)
            self.objectdetector = 'ssd'

            assert feature_extractor.lower(
            ) == 'vgg16', 'Only VGG16 feature extractor is supported for now'
            from cral.models.object_detection.SSD import SSDLoss
            from cral.models.object_detection.SSD import create_ssd_model
            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn, self.predictor_sizes = create_ssd_model(  # noqa: E501
                    config,
                    feature_extractor=feature_extractor,
                    num_classes=num_classes,
                    weights=weights)

            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'object_detection_meta' in dic.keys():
                            if 'config' in dic['object_detection_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['object_detection_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'object_detection_meta'].keys():
                                old_extractor = dic['object_detection_meta'][
                                    'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                elif not (config.check_equality(old_config)):
                    assert False, 'Weights could not be loaded'
                # preprocessing fn to be checked what exactly does it say
                self.model, self.preprocessing_fn, self.predictor_sizes = create_ssd_model(  # noqa: E501
                    config,
                    feature_extractor=feature_extractor,
                    num_classes=num_classes,
                    weights=None)
                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'
            loss = SSDLoss(
                neg_pos_ratio=config.neg_pos_ratio, alpha=config.alpha)
            loss = loss.compute_loss

            architecture = 'ssd'

        elif isinstance(config, RetinanetConfig):
            log_retinanet_config_params(config)

            self.objectdetector = 'retinanet'

            from cral.models.object_detection.retinanet import (focal,
                                                                smooth_l1)
            if feature_extractor == 'detnet':
                print(
                    'For `detnet` backbone Pyramid_levels should be [4, 4, 4, 5, 6] '  # noqa: E501
                )
                print('Changing pyramid levels to supported size')
                config.pyramid_levels = [4, 4, 4, 5, 6]

            if weights in ('imagenet', None):
                self.model, self.preprocessing_fn = get_retinanet_fromconfig(
                    feature_extractor=feature_extractor,
                    weights=weights,
                    config=config,
                    num_classes=num_classes,
                    base_trainable=base_trainable)
            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                old_cral_path = os.path.join(weights, 'assets', 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'object_detection_meta' in dic.keys():
                            if 'config' in dic['object_detection_meta'].keys():
                                old_config = jsonpickle.decode(
                                    dic['object_detection_meta']['config'])
                            if 'feature_extractor' in dic[
                                    'object_detection_meta'].keys():
                                old_extractor = dic['object_detection_meta'][
                                    'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch {feature_extractor}!={old_extractor}'  # noqa: E501
                elif not (config.check_equality(old_config)):
                    assert False, 'Weights could not be loaded'

                self.model, self.preprocessing_fn = get_retinanet_fromconfig(
                    feature_extractor=feature_extractor,
                    weights=None,
                    config=config,
                    num_classes=num_classes,
                    base_trainable=base_trainable)

                self.model.load_weights(
                    os.path.join(weights, 'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            # retinanet default losses
            loss = {'regression': smooth_l1(), 'classification': focal()}

            architecture = 'retinanet'

        elif isinstance(config, FasterRCNNConfig):
            with graph_mode():
                assert not tf.executing_eagerly()
                # from cral.models.object_detection import log_FasterRCNN_config_params  # noqa: E501
                from cral.models.object_detection import create_FasterRCNN

                assert isinstance(config, FasterRCNNConfig), 'please provide a `FasterRCNNConfig()` object'  # noqa: E501

#               config['NUM_ClASSES'] = num_classes
#               print(config.NUM_ClASSES)
                num_classes = int(self.cral_meta_data['num_classes'])
                print(num_classes)
                log_FasterRCNN_config_params(config)
                optimizer_frcnn = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, clipnorm=5.0)  # noqa: E501

                if weights in ('imagenet', None):
                    self.model = create_FasterRCNN(feature_extractor,
                                                   config, num_classes,
                                                   weights, base_trainable,
                                                   mode='training')

                elif tf.saved_model.contains_saved_model(weights):
                    print('\nLoading Weights\n')
                    old_config = None
                    old_extractor = None
                    segmind_cral_path = os.path.dirname(weights)
                    old_cral_path = os.path.join(segmind_cral_path,
                                                 'segmind.cral')
                    if os.path.isfile(old_cral_path):
                        with open(old_cral_path) as old_cral_file:
                            dic = json.load(old_cral_file)

                            if 'object_detection_meta' in dic.keys():
                                if 'config' in dic['object_detection_meta'].keys():  # noqa: E501
                                    old_config = jsonpickle.decode(
                                      dic['object_detection_meta']['config'])
                                if 'feature_extractor' in dic[
                                  'object_detection_meta'].keys():
                                    old_extractor = dic['object_detection_meta'][  # noqa: E501
                                      'feature_extractor']

                    if None in (old_extractor, old_config):
                        assert False, 'Weights file is not supported'
                    elif feature_extractor != old_extractor:
                        assert False, f'feature_extractor mismatch {feature_extractor} != {old_extractor}'  # noqa: E501
                    # elif not (config.check_equality(old_config)):
                    elif vars(config) != vars(old_config):
                        assert False, 'Weights could not be loaded'

                    self.model = create_FasterRCNN(feature_extractor,
                                                   config, num_classes,
                                                   None, base_trainable,
                                                   mode='training')

                    self.model.load_weights(
                      os.path.join(weights, 'variables', 'variables'))
                else:
                    assert False, 'Weights file is not supported'

                loss_names = ['rpn_class_loss', 'rpn_bbox_loss',
                              'rcnn_class_loss', 'rcnn_bbox_loss']
                for name in loss_names:
                    layer = self.model.get_layer(name)
                    loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))  # noqa: E501
                    self.model.add_loss(loss)
                reg_losses = [tf.keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32) for w in self.model.trainable_weights if 'gamma' not in w.name and 'beta' not in w.name]  # noqa: E501
                self.model.add_loss(tf.add_n(reg_losses))
                print('Losses added')
                architecture = 'FasterRCNN'

        else:
            raise ValueError('argument to `config` is not understood.')

        if isinstance(config, FasterRCNNConfig):
            with graph_mode():
                assert not tf.executing_eagerly()
                # Model parallelism
                if distribute_strategy is None:
                    self.model.compile(optimizer=optimizer_frcnn, loss=[None] * len(self.model.outputs))  # noqa: E501
                else:
                    with distribute_strategy.scope():
                        self.model.compile(optimizer=optimizer_frcnn, loss=[None] * len(self.model.outputs))  # noqa: E501
                print('model compiled')
                for name in loss_names:
                    layer = self.model.get_layer(name)
                    self.model.metrics_names.append(name)
                    loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))  # noqa: E501
                    self.model.add_metric(loss, name=name, aggregation='mean')
                print('Metrics added')

        else:
            # custom image normalizing function
            if preprocessing_fn is not None:
                self.preprocessing_fn = preprocessing_fn

            @tf.function(input_signature=[tf.TensorSpec([None, None, 3], dtype=tf.float32)])  # noqa: E501
            def _preprocess(image_array):
                """tf.function-deocrated version of preprocess_"""
                im_arr = self.preprocessing_fn(image_array)
                input_batch = tf.expand_dims(im_arr, axis=0)
                return input_batch

            # Attach function to Model
            self.model.preprocess = _preprocess

            # Model parallelism
            if distribute_strategy is None:
                self.model.compile(loss=loss, optimizer=optimizer)
            else:
                with distribute_strategy.scope():
                    self.model.compile(loss=loss, optimizer=optimizer)

        algo_meta = dict(
            feature_extractor=feature_extractor,
            architecture=architecture,
            weights=weights,
            base_trainable=base_trainable,
            config=jsonpickle.encode(config))

        object_detection_meta = dict(object_detection_meta=algo_meta)

        self.update_project_file(object_detection_meta)

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
        # return

        assert isinstance(num_epochs,
                          int), 'num epochs to run should be in `int`'
        assert os.path.isdir(snapshot_path), '{} doesnot exist'.format(
            snapshot_path)
        assert isinstance(callbacks, list)
        assert isinstance(validate_every_n, int)

        snapshot_prefix = str(snapshot_prefix)

        if validation_batch_size is None:
            validation_batch_size = batch_size

        num_classes = int(self.cral_meta_data['num_classes'])
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
        from cral.augmentations.engine import ObjectDetection as \
            objectdetection_augmentor

        if self.cral_meta_data['object_detection_meta'][
                'architecture'] == 'yolo':
            from cral.models.object_detection.YoloV3 import YoloGenerator
            yolo_config = jsonpickle.decode(
                self.cral_meta_data['object_detection_meta']['config'])
            assert isinstance(
                yolo_config, YoloV3Config
            ), 'Expected an instance of cral.models.object_detection.YoloV3'

            if self.aug_pipeline is not None:
                augmentation = objectdetection_augmentor(
                    self.aug_pipeline,
                    annotation_format=yolo_config.input_anno_format)
            else:
                augmentation = None

            data_gen = YoloGenerator(
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                num_classes=num_classes,
                config=yolo_config,
                batch_size=batch_size,
                preprocessing_fn=self.preprocessing_fn)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:
                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size / validation_batch_size
            else:
                test_input_function = None
                validation_steps = None

        if self.cral_meta_data['object_detection_meta'][
                'architecture'] == 'ssd':
            from cral.models.object_detection.SSD import SSD300Generator
            ssd_config = jsonpickle.decode(
                self.cral_meta_data['object_detection_meta']['config'])
            assert isinstance(
                ssd_config, SSD300Config
            ), 'Expected an instance of cral.models.object_detection.SSD300'

            if self.aug_pipeline is not None:
                augmentation = objectdetection_augmentor(
                    self.aug_pipeline,
                    annotation_format=ssd_config.input_anno_format)
            else:
                augmentation = None

            parser = SSD300Generator(
                config=ssd_config,
                predictor_sizes=self.predictor_sizes,
                num_classes=num_classes,
                batch_size=batch_size,
                preprocess_input=self.preprocessing_fn)

            train_input_function = parser.parse_tfrecords(
                filename=train_tfrecords)

            if test_set_size > 0:
                test_input_function = parser.parse_tfrecords(
                    filename=test_tfrecords)
                validation_steps = test_set_size / validation_batch_size
            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['object_detection_meta'][
                'architecture'] == 'retinanet':

            from cral.models.object_detection import RetinanetGenerator

            retinanet_config = jsonpickle.decode(
                self.cral_meta_data['object_detection_meta']['config'])

            assert isinstance(
                retinanet_config, RetinanetConfig
            ), 'Expected an instance of cral.models.object_detection.RetinanetConfig'  # noqa: E501

            if self.aug_pipeline is not None:
                augmentation = objectdetection_augmentor(
                    self.aug_pipeline,
                    annotation_format=retinanet_config.input_anno_format)
            else:
                augmentation = None

            data_gen = RetinanetGenerator(
                config=retinanet_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                min_side=retinanet_config.min_side,
                max_side=retinanet_config.max_side,
                num_classes=num_classes,
                bboxes_format=self.cral_meta_data['annotation_format'],
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size)

            train_input_function = data_gen.get_train_function()

            if test_set_size > 0:

                test_input_function = data_gen.get_test_function()
                validation_steps = test_set_size / validation_batch_size

            else:
                test_input_function = None
                validation_steps = None

        elif self.cral_meta_data['object_detection_meta'][
          'architecture'] == 'FasterRCNN':
            with graph_mode():
                assert not tf.executing_eagerly()
                from cral.models.object_detection import FasterRCNNGenerator

                fasterrcnn_config = jsonpickle.decode(
                  self.cral_meta_data['object_detection_meta']['config'])

                assert isinstance(fasterrcnn_config, FasterRCNNConfig), 'Expected an instance of cral.models.object_detection.FasterRCNNConfig'  # noqa: E501

                data_gen = FasterRCNNGenerator(
                    config=fasterrcnn_config,
                    train_tfrecords=train_tfrecords,
                    test_tfrecords=test_tfrecords,
                    # processing_func=self.preprocessing_fn,
                    # augmentation=augmentation,
                    batch_size=batch_size, num_classes=num_classes)

                train_input_function = data_gen.get_train_function()

                if test_set_size > 0:

                    test_input_function = data_gen.get_test_function()
                    validation_steps = test_set_size // validation_batch_size

                else:
                    test_input_function = None
                    validation_steps = None

        if steps_per_epoch is None:
            steps_per_epoch = training_set_size / batch_size

        # callbacks.append(KerasCallback(log_evry_n_step))
        # callbacks.append(KerasCallback())
        # callbacks.append(checkpoint_callback(
        #     snapshot_every_epoch=snapshot_every_n,
        #     snapshot_path=snapshot_path,
        #     checkpoint_prefix=snapshot_prefix,
        #     save_h5=False))

        # Attach segmind.cral as an asset
        self.cral_file_path = os.path.join(snapshot_path, 'segmind.cral')
        tf.io.gfile.copy(self.cral_file, self.cral_file_path, overwrite=True)
        cral_asset_file = tf.saved_model.Asset('segmind.cral')

        directory = os.getcwd()
        self.cral_path = os.path.join(directory, 'segmind.cral')
        tf.io.gfile.copy(self.cral_file, self.cral_path, overwrite=True)

        self.model.cral_file = cral_asset_file

        # log_param('training_steps_per_epoch', int(steps_per_epoch))
        # if test_set_size>0:
        #     log_param('val_steps_per_epoch', int(validation_steps))
        if self.cral_meta_data['object_detection_meta'][
          'architecture'] == 'FasterRCNN':
            with graph_mode():
                assert not tf.executing_eagerly()
                self.model.fit(
                    x=train_input_function,
                    epochs=num_epochs,
                    callbacks=callbacks,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_input_function,
                    validation_steps=validation_steps,
                    validation_freq=validate_every_n)
        else:
            self.model.fit(
                x=train_input_function,
                epochs=num_epochs,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_data=test_input_function,
                validation_steps=validation_steps,
                validation_freq=validate_every_n)

        if self.cral_meta_data['object_detection_meta'][
          'architecture'] == 'FasterRCNN':
            with graph_mode():
                final_weights_path = os.path.join(
                  snapshot_path, str(snapshot_prefix)+'_final.h5')
                self.model.save_weights(filepath=final_weights_path,
                                        overwrite=True)
                print('Saved the final weights to :\n {}'.format(final_weights_path))  # noqa: E501
        else:
            final_model_path = os.path.join(snapshot_path, str(snapshot_prefix)+'_final')  # noqa: E501

            self.model.save(
                  filepath=final_model_path,
                  overwrite=True)

            print('Saved the final Model to :\n {}'.format(final_model_path))

    def prediction_model(self, checkpoint_file):

        try:
            location_to_cral_file = self.model.cral_file.asset_path.numpy()
            with open(location_to_cral_file) as f:
                metainfo = json.loads(f.read())

        except AttributeError:
            print(
                "Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted"  # noqa: E501
            )

        architecture = metainfo['object_detection_meta']['architecture']
        num_classes = int(metainfo['num_classes'])
        feature_extractor = metainfo['object_detection_meta'][
            'feature_extractor']

        if architecture == 'retinanet':

            # from cral.models.object_detection.retinanet.predict_script import (  # noqa: E501
            #     load_model, RetinanetPredictor)
            from cral.models.object_detection.retinanet.base import \
                get_prediction_model
            self.model = keras.models.load_model(checkpoint_file, compile=False)  # noqa: E501
            retinanet_config = jsonpickle.decode(
                metainfo['object_detection_meta']['config'])
            assert isinstance(
                retinanet_config, RetinanetConfig
            ), 'Expected an instance of cral.models.object_detection.RetinanetConfig'  # noqa: E501

            training_model = self.model

            _, preprocessing_fn = classification_networks[feature_extractor](
                weights=None)

            if sorted(training_model.output_names) == [
                    'classification', 'regression'
            ]:
                print('converting ...')
                prediction_model = get_prediction_model(
                    model=training_model, config=retinanet_config)

            else:
                prediction_model = training_model

            pred_obj = RetinanetPredictor_temp(
                min_side=retinanet_config.min_side,
                max_side=retinanet_config.max_side,
                model=prediction_model,
                preprocessing_func=preprocessing_fn)

            return pred_obj.predict

        elif architecture == 'yolo':
            from cral.models.object_detection.YoloV3.predict import (
                freeze_model, detect_image)
            self.model = keras.models.load_model(checkpoint_file, compile=False)  # noqa: E501
            yolo_config = jsonpickle.decode(
                metainfo['object_detection_meta']['config'])
            assert isinstance(
                yolo_config, YoloV3Config
            ), 'Expected an instance of cral.models.object_detection.YoloV3Config'  # noqa: E501

            _, preprocessing_fn = classification_networks[feature_extractor](
                weights=None)
            p_model = freeze_model(
                checkpoint_file, yolo_config, num_classes=num_classes)

            def prediction_func(img_path):
                image_array = np.array(Image.open(img_path))
                boxes_f = []
                bboxes, scores, labels = detect_image(
                    model=p_model, image=image_array, config=yolo_config)
                for box in bboxes:
                    top, left, bottom, right = box

                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image_array.shape[1],
                                 np.floor(bottom + 0.5).astype('int32'))
                    right = min(image_array.shape[0],
                                np.floor(right + 0.5).astype('int32'))

                    box = (left, top, right, bottom)
                    boxes_f.append(box)

                return boxes_f, scores, labels

            return prediction_func

        elif architecture == 'ssd':
            from tensorflow.keras.preprocessing.image import load_img
            self.model = keras.models.load_model(checkpoint_file, compile=False)  # noqa: E501
            ssd_config = jsonpickle.decode(
                metainfo['object_detection_meta']['config'])
            assert isinstance(
                ssd_config, SSD300Config
            ), 'Expected an instance of cral.models.object_detection.SSD300Config'  # noqa: E501
            from tensorflow.keras.applications.vgg16 import preprocess_input
            model = decode_detections(
                self.model,
                ssd_config,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

            def prediction_func(img_path):
                input_images = []
                pil_input_images = []
                # original_image_list = []
                pil_image = load_img(
                    img_path,
                    target_size=(ssd_config.height, ssd_config.width))
                original_image = load_img(img_path)
                h = original_image.height
                w = original_image.width

                pil_input_images.append(pil_image)
                img = np.array(pil_image)
                input_images.append(img)
                input_images = preprocess_input(np.array(input_images))

                bboxes, scores, labels = model.predict(input_images)
                bbox = bboxes[0].copy()
                bbox[:, 0] = bbox[:, 0] * (h / ssd_config.height)
                bbox[:, 1] = bbox[:, 1] * (w / ssd_config.width)
                bbox[:, 2] = bbox[:, 2] * (h / ssd_config.height)
                bbox[:, 3] = bbox[:, 3] * (w / ssd_config.width)
                return bbox, scores.flatten(), labels.flatten()

            return prediction_func

        elif architecture == 'FasterRCNN':

            from cral.models.object_detection import create_FasterRCNN
            try:
                location_to_cral_file = os.path.join(
                  os.path.dirname(checkpoint_file), 'segmind.cral')
                with open(location_to_cral_file) as f:
                    metainfo = json.loads(f.read())
                print(metainfo)

            except AttributeError:
                print("Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted")  # noqa: E501

            architecture = metainfo['object_detection_meta']['architecture']
            num_classes = int(metainfo['num_classes'])
            feature_extractor = metainfo['object_detection_meta'][
              'feature_extractor']
            weights = 'imagenet'
            fasterrcnn_config = jsonpickle.decode(
              metainfo['object_detection_meta']['config'])

            if isinstance(fasterrcnn_config, FasterRCNNConfig):
                assert isinstance(fasterrcnn_config, FasterRCNNConfig), 'Please provide a `FasterRCNNConfig()` object.'  # noqa: E501

            if feature_extractor not in ['resnet50', 'resnet101']:
                raise ValueError(f'{feature_extractor} not yet supported ..')
            self.prediction_model = create_FasterRCNN(feature_extractor,
                                                      fasterrcnn_config,
                                                      num_classes, weights,
                                                      False,
                                                      mode='inference')
            self.prediction_model.load_weights(checkpoint_file,
                                               by_name=True,
                                               skip_mismatch=True)

            from cral.models.object_detection import FasterRCNNPredictor
            pred_object = FasterRCNNPredictor(
                height=fasterrcnn_config.height,
                width=fasterrcnn_config.width,
                model=self.prediction_model,
                num_classes=num_classes,
                config=fasterrcnn_config)

            return pred_object.predict

        else:
            print('{} Not yet supported'.format(architecture))


# TODO integrate this into the pipeline correctly
class RetinanetPredictor_temp():
    """docstring for RetinanetPredictor."""

    def __init__(self, min_side, max_side, model, preprocessing_func):
        # super(RetinanetPredictor, self).__init__(*args, **kwargs)
        self.min_side = min_side
        self.max_side = max_side
        self.model = model
        self.preprocessing_func = preprocessing_func

    def load_image(self, image_path):
        img_array = np.array(
            keras.preprocessing.image.load_img(path=image_path))
        # print(img_array.shape,img_array.dtype)
        return img_array

    def predict(self, image):
        im = self.load_image(image)
        # print(im.shape)

        smallest_side = min(im.shape[0], im.shape[1])
        largest_side = max(im.shape[0], im.shape[1])

        scale = self.min_side / smallest_side

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        image = cv2.resize(im, None, fx=scale, fy=scale)
        image = np.expand_dims(image, axis=0)

        images_batch = self.preprocessing_func(image)
        images_batch = tf.cast(images_batch, tf.keras.backend.floatx())
        # print(images_batch.shape, images_batch.dtype)

        bboxes, confidence, label = self.model.predict(images_batch)

        return bboxes[0].astype(int) / scale, confidence[0], label[0]
