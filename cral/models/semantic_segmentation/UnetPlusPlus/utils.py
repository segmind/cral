import cv2
import numpy as np
import tensorflow as tf
from cral.models.semantic_segmentation.utils import do_crf  # noqa: F401
# from segmind import log_params
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dropout, InputSpec, Layer


class Expand_Dims(Layer):
    """docstring for Expand_Dims."""

    def call(self, inputs):
        outputs = K.expand_dims(inputs, 1)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], 1] + input_shape[1:])


class Upsample(Layer):
    """Image Upsample layer.

    Resize the batched image input to target height and width. The input
    should be a 4-D tensor in the format of NHWC.
    Arguments:
    height: Integer, the height of the output shape.
    width: Integer, the width of the output shape.
    interpolation: String, the interpolation method. Defaults to `bilinear`.
      Supports `bilinear`, `nearest`, `bicubic`, `area`, `lanczos3`,
      `lanczos5`, `gaussian`, `mitchellcubic`
    name: A string, the name of the layer.
    """

    def __init__(
            self,
            height,
            width,
            interpolation='bilinear',
            # name='Upsample',
            **kwargs):
        self.target_height = height
        self.target_width = width
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)
        super(Upsample, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = tf.image.resize(
            images=inputs,
            size=[self.target_height, self.target_width],
            method=self.interpolation)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([
            input_shape[0], self.target_height, self.target_width,
            input_shape[3]
        ])

    def get_config(self):
        config = {
            'height': self.target_height,
            'width': self.target_width,
            'interpolation': self.interpolation,
        }
        base_config = super(Upsample, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    act = 'relu'
    dropout_rate = 0.5
    x = Conv2D(
        nb_filter, (kernel_size, kernel_size),
        activation=act,
        name='conv' + stage + '_1',
        padding='same')(
            input_tensor)
    x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(
        nb_filter, (kernel_size, kernel_size),
        activation=act,
        name='conv' + stage + '_2',
        padding='same')(
            x)
    x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
    return x


class UnetPlusPlusConfig(object):
    """docstring for Deeplabv3Config."""

    def __init__(self,
                 height=320,
                 width=320,
                 num_upsample_layers=5,
                 filters=[64, 128, 256, 512, 1024],
                 deep_supervision=True):
        self.height = height
        self.width = width
        self.num_upsample_layers = num_upsample_layers
        self.filters = filters
        self.deep_supervision = deep_supervision
        assert len(
            filters
        ) >= num_upsample_layers, 'filters should be a list with length >= num_upsample_layers'  # noqa: E501
        assert num_upsample_layers in [
            2, 3, 4, 5
        ], 'num_upsample_layers should be in [2, 3, 4, 5]'
        assert height % 32 == 0 and width % 32 == 0, 'height and width should be multiple of 32'  # noqa: E501
        self.input_shape = (self.height, self.width, 3)


def log_UnetPlusPlus_config_params(config):

    assert isinstance(
        config, UnetPlusPlusConfig), 'config not supported {}'.format(config)
    # config_data = vars(config)
    # log_params(config_data)


class UnetPlusPlusPredictor(object):
    """docstring for Deeplabv3Predictor."""

    def __init__(self, height, width, model, preprocessing_func, dcrf):
        # super(RetinanetPredictor, self).__init__(*args, **kwargs)
        self.height = height
        self.width = width
        self.model = model
        self.preprocessing_func = preprocessing_func
        self.allow_dcrf = dcrf

    def load_image(self, image_path):
        img_array = np.array(
            tf.keras.preprocessing.image.load_img(path=image_path))
        return img_array

    def predict(self, image):
        im = self.load_image(image)

        image = cv2.resize(im, (self.width, self.height))
        image_array = np.expand_dims(image, axis=0)

        images_batch = self.preprocessing_func(image_array)

        images_batch = tf.cast(images_batch, tf.keras.backend.floatx())

        y = self.model.predict(images_batch)[0]
        y = np.argmax(y, axis=-1)

        return y
