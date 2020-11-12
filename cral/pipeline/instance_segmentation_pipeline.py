import json
import os

import jsonpickle
import tensorflow as tf

from cral.callbacks import checkpoint_callback
# from segmind import KerasCallback, log_param
from cral.common import log_gpu_params
from cral.data_feeder.instance_seg_data_feeder import \
    create_tfrecords as create_tfrecords_instance_segmentation
from cral.data_versioning import instanceSeg_dataset_hasher
from cral.models.instance_segmentation.MaskRCNN import (MaskRCNNConfig,
                                                        log_MaskRCNN_config_params)  # noqa: E501
from cral.pipeline.core import PipelineBase
from cral.tracking import KerasCallback, log_param


class InstanceSegPipe(PipelineBase):
    """Cral pipeline for classification task."""
    def __init__(self, *args, **kwargs):
        super(InstanceSegPipe, self).__init__(
          task_type='instance_segmentation', *args, **kwargs)
        self.cral_file_path = ''

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
            split (float, optional): float to divide training dataset
                                     into traing and val
            img_to_anno (function, optional): Function to convert image name to
                                              annotation name
        """
        self.dataset_hash, self.dataset_csv_path, self.dataset_json = instanceSeg_dataset_hasher(  # noqa: E501
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
        print(self.dataset_hash, self.dataset_csv_path, self.dataset_json)

    def set_aug(self):
        pass

    def visualize_data(self):
        pass

    def lock_data(self):
        """Parse Data and makes tf-records and creates meta-data."""
        meta_info = create_tfrecords_instance_segmentation(self.dataset_json,
                                                           self.dataset_csv_path)  # noqa: E501

        self.update_project_file(meta_info)
        return meta_info

    def set_algo(self,
                 feature_extractor,
                 config,
                 weights='imagenet',
                 base_trainable=False,
                 preprocessing_fn=None,
                 optimizer=tf.keras.optimizers.SGD(lr=1e-4, clipnorm=5.0),
                 distribute_strategy=None):

        num_classes = int(self.cral_meta_data['num_classes'])
        architecture = None

        self.preprocessing_fn = None

        feature_extractor = feature_extractor.lower()

        if isinstance(config, MaskRCNNConfig):
            from cral.models.instance_segmentation import (  # noqa: E501
                create_MaskRCNN, log_MaskRCNN_config_params)

            assert isinstance(config, MaskRCNNConfig), 'please provide a `MaskRCNNConfig()` object'  # noqa: E501

            num_classes = int(self.cral_meta_data['num_classes'])
            print(num_classes)
            log_MaskRCNN_config_params(config)

            if weights in ('imagenet', None):

                self.model = create_MaskRCNN(feature_extractor, config,
                                             num_classes, weights,
                                             base_trainable, mode='training')

            elif tf.saved_model.contains_saved_model(weights):
                print('\nLoading Weights\n')
                old_config = None
                old_extractor = None
                segmind_cral_path = os.path.dirname(weights)
                old_cral_path = os.path.join(segmind_cral_path, 'segmind.cral')
                if os.path.isfile(old_cral_path):
                    with open(old_cral_path) as old_cral_file:
                        dic = json.load(old_cral_file)

                        if 'instance_segmentation_meta' in dic.keys():
                            if 'config' in dic[
                              'instance_segmentation_meta'].keys():
                                old_config = jsonpickle.decode(
                                  dic['instance_segmentation_meta']['config'])
                            if 'feature_extractor' in dic[
                              'instance_segmentation_meta'].keys():
                                old_extractor = dic['instance_segmentation_meta'][  # noqa: E501
                                  'feature_extractor']

                if None in (old_extractor, old_config):
                    assert False, 'Weights file is not supported'
                elif feature_extractor != old_extractor:
                    assert False, f'feature_extractor mismatch \
                    {feature_extractor} != {old_extractor}'
                # elif not (config.check_equality(old_config)):
                elif vars(config) != vars(old_config):
                    assert False, 'Weights could not be loaded'

                self.model = create_MaskRCNN(feature_extractor, config,
                                             num_classes, None,
                                             base_trainable, mode='training')

                self.model.load_weights(os.path.join(weights,
                                                     'variables', 'variables'))
            else:
                assert False, 'Weights file is not supported'

            loss_names = ['rpn_class_loss', 'rpn_bbox_loss',
                          'mrcnn_class_loss', 'mrcnn_bbox_loss',
                          'mrcnn_mask_loss']
            for name in loss_names:
                layer = self.model.get_layer(name)
                loss = (tf.reduce_mean(
                  input_tensor=layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))  # noqa: E501
                self.model.add_loss(loss)
            reg_losses = [tf.keras.regularizers.l2(
                          config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)  # noqa: E501
                          for w in self.model.trainable_weights
                          if 'gamma' not in w.name and 'beta' not in w.name]
            self.model.add_loss(tf.add_n(reg_losses))
            print('Losses added')
            architecture = 'MaskRCNN'

        else:
            raise ValueError('argument to `config` is not understood.')

        # custom image normalizing function
        if preprocessing_fn is not None:
            self.preprocessing_fn = preprocessing_fn

        # Model parallelism
        if distribute_strategy is None:
            self.model.compile(optimizer=optimizer,
                               loss=[None] * len(self.model.outputs))
        else:
            with distribute_strategy.scope():
                self.model.compile(optimizer=optimizer,
                                   loss=[None] * len(self.model.outputs))

        for name in loss_names:
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) * config.LOSS_WEIGHTS.get(name, 1.))  # noqa: E501
            self.model.add_metric(loss, name=name, aggregation='mean')
        print('Metrics added')

        algo_meta = dict(
            feature_extractor=feature_extractor,
            architecture=architecture,
            weights=weights,
            base_trainable=base_trainable,
            config=jsonpickle.encode(config))

        instance_segmentation_meta = dict(instance_segmentation_meta=algo_meta)

        self.update_project_file(instance_segmentation_meta)

#     def visualize_data(self):
#         pass

    def train(self,
              num_epochs,
              snapshot_prefix,
              snapshot_path,
              snapshot_every_n,
              batch_size=1,
              validation_batch_size=None,
              validate_every_n=1,
              callbacks=[],
              steps_per_epoch=None,
              compile_options=None,
              distribute_strategy=None,
              log_evry_n_step=100):

        assert isinstance(num_epochs, int), 'num epochs to run should be in `int`'  # noqa: E501
        assert isinstance(callbacks, list)
        assert isinstance(validate_every_n, int)

        snapshot_prefix = str(snapshot_prefix)
        batch_size = 1
        if validation_batch_size is None:
            validation_batch_size = batch_size

        num_classes = int(self.cral_meta_data['num_classes'])
        training_set_size = int(self.cral_meta_data['num_training_images'])
        test_set_size = int(self.cral_meta_data['num_test_images'])

        if self.model is None:
            raise ValueError('please define a model first using set_algo() function')  # noqa: E501

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

        if self.cral_meta_data['instance_segmentation_meta'][
          'architecture'] == 'MaskRCNN':

            from cral.models.instance_segmentation import MaskRCNNGenerator

            maskrcnn_config = jsonpickle.decode(
              self.cral_meta_data['instance_segmentation_meta']['config'])

            assert isinstance(
              maskrcnn_config, MaskRCNNConfig), 'Expected an instance of cral.models.instance_segmentation.MaskRCNNConfig'  # noqa: E501

            augmentation = self.aug_pipeline

            data_gen = MaskRCNNGenerator(
                config=maskrcnn_config,
                train_tfrecords=train_tfrecords,
                test_tfrecords=test_tfrecords,
                processing_func=self.preprocessing_fn,
                augmentation=augmentation,
                batch_size=batch_size, num_classes=num_classes)

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
            steps_per_epoch = training_set_size//batch_size

        # callbacks.append(KerasCallback(log_evry_n_step))
        callbacks.append(KerasCallback())
        callbacks.append(checkpoint_callback(
            snapshot_every_epoch=snapshot_every_n,
            snapshot_path=snapshot_path,
            checkpoint_prefix=snapshot_prefix,
            save_h5=True))

        # Attach segmind.cral as an asset
        self.cral_file_path = os.path.join(snapshot_path, 'segmind.cral')
        tf.io.gfile.copy(self.cral_file, self.cral_file_path, overwrite=True)

        directory = os.getcwd()
        self.cral_path = os.path.join(directory, 'segmind.cral')
        tf.io.gfile.copy(self.cral_file, self.cral_path, overwrite=True)
        # cral_asset_file = tf.saved_model.Asset('cral_file_path')

        # self.model.cral_file = self.cral_asset_file

        log_param('training_steps_per_epoch', int(steps_per_epoch))
        if test_set_size > 0:
            log_param('val_steps_per_epoch', int(validation_steps))

        log_gpu_params()

        self.model.fit(
            x=train_input_function,
            epochs=num_epochs,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_input_function,
            validation_steps=validation_steps,
            validation_freq=validate_every_n)

        final_weights_path = os.path.join(snapshot_path,
                                          str(snapshot_prefix)+'_final.h5')

        self.model.save_weights(filepath=final_weights_path, overwrite=True)
        print('Saved the final weights to :\n {}'.format(final_weights_path))

    def prediction_model(self, checkpoint_file):

        from cral.models.instance_segmentation import create_MaskRCNN
        try:
            location_to_cral_file = os.path.join(
              os.path.dirname(checkpoint_file), 'segmind.cral')
            with open(location_to_cral_file) as f:
                metainfo = json.loads(f.read())
            print(metainfo)

        except AttributeError:
            print("Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted")  # noqa: E501

        architecture = metainfo['instance_segmentation_meta']['architecture']
        num_classes = int(metainfo['num_classes'])
        feature_extractor = metainfo['instance_segmentation_meta'][
          'feature_extractor']
        weights = 'imagenet'
        maskrcnn_config = jsonpickle.decode(
          metainfo['instance_segmentation_meta']['config'])
        if isinstance(maskrcnn_config, MaskRCNNConfig):
            assert isinstance(maskrcnn_config, MaskRCNNConfig), 'Please provide a `MaskRCNNConfig()` object.'  # noqa: E501
        if architecture == 'MaskRCNN':
            if feature_extractor not in ['resnet50', 'resnet101']:
                raise ValueError(f'{feature_extractor} not yet supported ..')
            self.prediction_model = create_MaskRCNN(feature_extractor,
                                                    maskrcnn_config,
                                                    num_classes, weights,
                                                    False, mode='inference')
            self.prediction_model.load_weights(checkpoint_file, by_name=True,
                                               skip_mismatch=True)

            from cral.models.instance_segmentation import MaskRCNNPredictor

            pred_object = MaskRCNNPredictor(
                height=maskrcnn_config.height,
                width=maskrcnn_config.width,
                model=self.prediction_model,
                config=maskrcnn_config)
            return pred_object.predict
