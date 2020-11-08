import cv2
import numpy as np
import tensorflow as tf
from cral.models.semantic_segmentation.utils import do_crf
# from cral.tracking import log_params
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D, InputSpec,
                                     Layer)


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

    Resize the batched image input to target height and width. The input should
    be a 4-D tensor in the format of NHWC.
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
        # self._interpolation_method = get_interpolation(interpolation)
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


class PspNetConfig(object):
    """docstring for Deeplabv3Config."""

    def __init__(self, height=576, width=576, down_sample_factor=8):
        self.height = height
        self.width = width
        self.down_sample_factor = down_sample_factor
        min_size = down_sample_factor * 6
        is_wrong_shape = (
            height % min_size != 0 or width % min_size != 0
            or height < min_size or width < min_size)
        assert down_sample_factor in [
            4, 8, 16
        ], 'Supported down_sample_factor -> 4, 8, 16'
        assert is_wrong_shape is False, 'height and width should be multiple of down_sample_factor*6'  # noqa: E501
        self.input_shape = (self.height, self.width, 3)


def log_PspNet_config_params(config):

    assert isinstance(config,
                      PspNetConfig), 'config not supported {}'.format(config)
    # config_data = vars(config)
    # log_params(config_data)


def pool_block(feats, pool_factor):
    h = K.int_shape(feats)[1]
    w = K.int_shape(feats)[2]
    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))
    ]

    x = AveragePooling2D(pool_size, strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Upsample(
        K.int_shape(x)[1] * strides[0],
        K.int_shape(x)[2] * strides[1])(
            x)
    return x


class PspNetPredictor(object):
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
        # print(img_array.shape,img_array.dtype)
        return img_array

    def predict(self, image):
        im = self.load_image(image)

        image = cv2.resize(im, (self.width, self.height))
        image_array = np.expand_dims(image, axis=0)

        images_batch = self.preprocessing_func(image_array)
        images_batch = tf.cast(images_batch, tf.keras.backend.floatx())
        # print(images_batch.shape, images_batch.dtype)

        y = self.model.predict(images_batch)[0]
        y = np.argmax(y, axis=-1)

        if self.allow_dcrf:
            # return densecrf(image.astype(np.uint8), y)
            return do_crf(image.astype(np.uint8), y)

        return y
