import json
import os
import tempfile
from abc import ABC, abstractmethod

import jsonpickle
import tensorflow as tf
from cral.augmentations.engine import Classification as AugmentorClassification
from cral.common import classification_networks
from cral.data_feeder.classification_parallel_data_feeder import \
    create_tfrecords as classification_tfrecord_creator
from cral.data_feeder.classification_utils import \
    parse_tfrecords as classification_tfrecord_parser
from cral.data_versioning.classification_data_parse_v2 import \
    make_csv as classification_dataset_hasher
# from cral.tracking.lite_extensions.client_utils import _get_experiment_id    # noqa: E501
# from cral.tracking import get_experiment, log_artifact, KerasCallback, log_param  # noqa: E501
from cral.models.classification import MLPConfig
from cral.models.classification.classification_utils import (
    ClassificationPredictor, densely_connected_head)
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

# from cral.common import log_gpu_params
# from cral.data_versioning import log_classification_dataset

# from cral.tracking.utils.autologging_utils import try_mlflow_log
# from .pipeline_utils import update_json as update_json
# from cral.utils import _PROJECTS_FOLDER

# from cral.callbacks import checkpoint_callback


class PipelineBase(ABC):
    """Parent abstract base class for pipelines of all task types.

    Attributes:
        aug_pipeline (TYPE): Description
        cral_file (TYPE): Description
        cral_meta_data (TYPE): Description
        exp_id (TYPE): Description
        model (TYPE): Description
        preprocessing_fn (TYPE): Description
        project_name (TYPE): Description
        task_type (TYPE): Description
        workspace (TYPE): Description
    """

    def __init__(self, task_type, workspace=tempfile.gettempdir()):
        self.workspace = workspace
        # self.exp_id = _get_experiment_id()
        self.exp_id = 'xxxyyyttt456575'
        # exp_info = get_experiment(self.exp_id)
        # self.project_name = exp_info.name.replace(self.exp_id.split('_')[0]+'_', '')  # noqa: E501
        self.project_name = 'demo'
        # Windows Fix
        # self.project_name = "|".join(exp_info.name.split("|")[1:])[1:]
        self.task_type = task_type
        self.model = None
        self.preprocessing_fn = None
        self.aug_pipeline = None

        self.create_project_file()

    def create_project_file(self, ):
        self.cral_meta_data = dict(
            workspace=self.workspace,
            exp_id=self.exp_id,
            project_name=self.project_name,
            task_type=self.task_type)

        self.cral_file = os.path.join('./', self.project_name + '.cral')

        with open(self.cral_file, 'w') as f:
            f.write(json.dumps(self.cral_meta_data, indent=2))

    def update_project_file(self, data_dict):
        assert isinstance(data_dict,
                          dict), '{} is not a dictionary'.format(data_dict)
        self.cral_meta_data.update(**data_dict)

        with open(self.cral_file, 'w') as f:
            f.write(json.dumps(self.cral_meta_data, indent=2))

    # @abstractmethod
    def import_project(self, cral_file):
        assert cral_file.endswith(
            '.cral'), 'only file ending with `.cral` is supported'
        with open(cral_file) as f:
            self.cral_meta_data = json.loads(f.read())
        self.cral_file = cral_file

    @abstractmethod
    def add_data(self):
        pass

    @abstractmethod
    def set_aug(self):
        pass

    @abstractmethod
    def lock_data(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass

    @abstractmethod
    def set_algo(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def prediction_model(self):
        pass


class ClassificationPipe(PipelineBase):
    """Cral pipeline for classification task."""

    def __init__(self, *args, **kwargs):
        super(ClassificationPipe, self).__init__(
            task_type='classification', *args, **kwargs)

    def add_data(self, *args, **kwargs):
        """Parses dataset once for generating metadata and versions the data.

        Args:
            *args: Description
            **kwargs: Description

        Deleted Parameters:
            train_images_dir (str): path to images
            val_images_dir (str, optional): path to validation images
            split (float, optional): float to divide training dataset into
                    training and validation
        """
        self.dataset_hash, self.dataset_csv_path, self.dataset_json = classification_dataset_hasher(  # noqa: E501
            tempfile.gettempdir(), *args, **kwargs)
        # try_mlflow_log(log_artifact, local_path=dataset_csv_path)
        # try_mlflow_log(log_artifact, local_path=dataset_json)

        with open(self.dataset_json) as f:
            self.data_dict = json.loads(f.read())
        self.update_project_file(self.data_dict)

    def set_aug(self, aug):
        """Sets the augmentation pipeline.

        Args:
            aug (TYPE): An albumentations data-augmentation pipeline
        """
        # Do a check on data
        self.aug_pipeline = AugmentorClassification(aug)
        # update_json(self)

    def visualize_data(self, image_url, allow_aug=False):
        """Summary.

        Args:
            image_url (TYPE): Description
            allow_aug (bool, optional): Description

        Raises:
            ValueError: Description
        """
        # tfrecord_dir = self.cral_meta_data['tfrecord_path']

        # train_tfrecords = list(
        #     glob.glob(os.path.join(tfrecord_dir, 'train*.tfrecord')))
        # test_tfrecords = list(
        #     glob.glob(os.path.join(tfrecord_dir, 'test*.tfrecord')))

        if allow_aug is True and self.aug_pipeline is None:
            raise ValueError('No augmentation pipeline has been provided')

    def lock_data(self, gen_stats=False):
        """Parse Data and makes tf-records and creates meta-data.

        Args:
            gen_stats (bool, optional): If True uses tfdv to create stats graph
        """
        meta_info = classification_tfrecord_creator(
            meta_json=os.path.join(tempfile.gettempdir(), 'dataset.json'),
            dataset_csv=os.path.join(tempfile.gettempdir(), 'dataset.csv'),
            out_path=tempfile.gettempdir())

        self.update_project_file(meta_info)

        # generate cavets overview html graphs with tfdv and log
        # disabling due to version problem
        if gen_stats:
            from cral.data_feeder.utils import generate_stats

            generate_stats(os.path.join(tempfile.gettempdir(), 'statistics'))

    def set_algo(self,
                 feature_extractor,
                 config,
                 weights='imagenet',
                 base_trainable=False,
                 preprocessing_fn=None,
                 optimizer=tf.keras.optimizers.Adam(lr=1e-4, clipnorm=0.001),
                 distribute_strategy=None):
        """Set model for training and prediction.

        Args:
            feature_extractor (str,model): Name of base model
            config: Is an instance of MLPConfig for the head of the network
            weights: one of `None` (random initialization),'imagenet'
                    (pre-training on ImageNet),or the path to the weights file
                    to be loaded
            base_trainable (bool, optional): If set False the base models
                    layers will not be trainable useful fortransfer learning
            preprocessing_fn (func, optional): needs to to be set if a in built
                    model is not being used
        Raises:
            ValueError: If network name assigned to `feature_extractor` is not
                    yet supported.
        """
        classification_algo_meta = dict(
            feature_extractor_from_cral=False, classification_meta=None)

        # if config is not None:
        assert isinstance(
            config, MLPConfig
        ), f'config has to be an object of MLPConfig but got{type(config)}'
        height = config.height
        width = config.width
        fully_connected_layer = config.fully_connected_layer
        dropout_rate = config.dropout_rate
        hidden_layer_activation = config.hidden_layer_activation
        final_layer_activation = config.final_layer_activation

        assert isinstance(feature_extractor,
                          str), 'expected a string got {} instead'.format(
                              type(feature_extractor))
        feature_extractor = feature_extractor.lower()
        if feature_extractor not in classification_networks.keys():
            raise ValueError('feature extractor has to be one of {}'.format(
                list(classification_networks.keys())))

        if weights in ('imagenet', None):
            backbone, self.preprocessing_fn = classification_networks[
                feature_extractor](
                    weights=weights, input_shape=(height, width, 3))

        elif tf.saved_model.contains_saved_model(weights):
            backbone, self.preprocessing_fn = classification_networks[
                feature_extractor](
                    weights=None, input_shape=(height, width, 3))

        else:
            assert False, 'Weights file is not supported'

        if preprocessing_fn is not None:
            # assert preprocessing function once
            self.preprocessing_fn = preprocessing_fn

        # freeze/train backbone
        backbone.trainable = base_trainable

        num_classes = self.cral_meta_data['num_classes']

        final_hidden_layer = densely_connected_head(
            feature_extractor_model=backbone,
            fully_connected_layer=fully_connected_layer,
            dropout_rate=dropout_rate,
            hidden_layer_Activation=hidden_layer_activation)

        output = Dense(
            units=num_classes, activation=final_layer_activation)(
                final_hidden_layer)

        # Assign resize dimensions
        resize_height = tf.constant(height, dtype=tf.int64)
        resize_width = tf.constant(width, dtype=tf.int64)

        @tf.function(
            input_signature=[tf.TensorSpec([None, None, 3], dtype=tf.uint8)])
        def _preprocess(image_array):
            """tf.function-deocrated version of preprocess_"""
            im_arr = tf.image.resize(image_array,
                                     (resize_height, resize_width))
            im_arr = self.preprocessing_fn(im_arr)
            input_batch = tf.expand_dims(im_arr, axis=0)
            return input_batch

        self.model = Model(
            inputs=backbone.inputs,
            outputs=output,
            name='{}_custom'.format(feature_extractor))

        if tf.saved_model.contains_saved_model(weights):
            self.model.load_weights(
                os.path.join(weights, 'variables', 'variables'))

        # Attach function to Model
        self.model.preprocess = _preprocess

        # Attach resize dimensions to Model
        self.model.resize_height = resize_height
        self.model.resize_width = resize_width

        # Model parallelism
        if distribute_strategy is None:
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
        else:
            with distribute_strategy.scope():
                self.model.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy'])

        classification_algo_meta['feature_extractor_from_cral'] = True

        classification_meta = dict(
            feature_extractor=feature_extractor,
            architecture='MLP',
            weights=weights,
            base_trainable=base_trainable,
            config=jsonpickle.encode(config))

        classification_algo_meta['classification_meta'] = classification_meta

        self.height = height
        self.width = width

        # log_param('feature_extractor', feature_extractor)

        self.update_project_file(classification_algo_meta)

        self.update_project_file(classification_algo_meta)

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
              log_evry_n_step=100):
        """This function starts the training loop, with metric logging enabled.

        Args:
            num_epochs (int): number of epochs to run training on
            snapshot_prefix (str): prefix to assign to the checkpoint file
            snapshot_path (str): a valid folder path where the checkpoints are
                    to be stored
            snapshot_every_n (int): take a snapshot at every nth epoch
            batch_size (int, optional): batch size
            validation_batch_size (None, optional): the batch size for
                    validation loop, if None(default) then equal to
                    `batch_size` argument
            validate_every_n (int, optional): Run validation every nth epoch
            callbacks (list, optional): list of keras callbacks to be passed to
                    model.fit() method
            steps_per_epoch (None, optional): steps size of each epoch
            compile_options (None, optional): A dictionary to be passed to
                    model.compile method

        Raises:
            ValueError: If model is not defined
        """

        assert isinstance(num_epochs,
                          int), 'num epochs to run should be in `int`'
        assert os.path.isdir(snapshot_path), '{} doesnot exist'.format(
            snapshot_path)
        assert isinstance(callbacks, list)
        assert isinstance(validate_every_n, int)

        snapshot_prefix = str(snapshot_prefix)

        if validation_batch_size is None:
            validation_batch_size = batch_size

        # self.height = height
        # self.width = width
        num_classes = int(self.cral_meta_data['num_classes'])
        training_set_size = int(self.cral_meta_data['num_training_images'])
        test_set_size = int(self.cral_meta_data['num_test_images'])

        if self.model is None:
            raise ValueError(
                'please define a model first using set_algo() function')

        if compile_options is not None:
            assert isinstance(compile_options, dict)
            self.model.compile(**compile_options)

        meta_info = dict(
            height=self.height,
            width=self.width,
            num_epochs=num_epochs,
            batch_size=batch_size)

        self.update_project_file(meta_info)

        tfrecord_dir = self.cral_meta_data['tfrecord_path']

        train_tfrecords = os.path.join(tfrecord_dir, 'train*.tfrecord')
        test_tfrecords = os.path.join(tfrecord_dir, 'test*.tfrecord')

        train_input_function = classification_tfrecord_parser(
            filenames=train_tfrecords,
            height=self.height,
            width=self.width,
            num_classes=num_classes,
            processing_func=self.preprocessing_fn,
            augmentation=self.aug_pipeline,
            batch_size=batch_size)

        if test_set_size > 0:

            test_input_function = classification_tfrecord_parser(
                filenames=test_tfrecords,
                height=self.height,
                width=self.width,
                num_classes=num_classes,
                processing_func=self.preprocessing_fn,
                augmentation=self.aug_pipeline,
                batch_size=validation_batch_size,
                num_repeat=-1)

            validation_steps = test_set_size / validation_batch_size

        else:
            test_input_function = None
            validation_steps = None

        if steps_per_epoch is None:
            steps_per_epoch = training_set_size / batch_size

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
        # pred_model = tf.keras.models.load_model('saved_model')
        # location_to_cral_file = pred_model.cral_file.asset_path.numpy()

        # log_param('training_steps_per_epoch', int(steps_per_epoch))
        # if test_set_size > 0:
        #     log_param('val_steps_per_epoch', int(validation_steps))
        # log_gpu_params()
        # Train & test
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

        self.model = keras.models.load_model(checkpoint_file, compile=False)

        try:
            location_to_cral_file = self.model.cral_file.asset_path.numpy()
            with open(location_to_cral_file) as f:
                metainfo = json.loads(f.read())

        except AttributeError:
            print(
                "Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted"  # noqa: E501
            )

        for k, v in metainfo.items():
            print(k, v)

        # architecture = metainfo['classification_meta']['architecture']
        # num_classes = int(metainfo['num_classes'])
        feature_extractor = metainfo['classification_meta'][
            'feature_extractor']
        size = (metainfo['height'], metainfo['width'])

        _, preprocessing_fn = classification_networks[feature_extractor](
            weights=None)

        pred_object = ClassificationPredictor(
            model=self.model, preprocessing_func=preprocessing_fn, size=size)

        return pred_object.predict
