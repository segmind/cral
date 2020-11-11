import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer

# from cral.tracking import log_params


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


class LinkNetConfig(object):
    """docstring for LinkNetConfig."""

    def __init__(self, height=576, width=576):

        # assert output_stride in [8,] #,16] <--- support to be added
        assert height % 32 == 0 and width % 32 == 0, 'Height and width both should be a multiple of 32'  # noqa: E501
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 3)


def log_LinkNet_config_params(config):

    assert isinstance(config,
                      LinkNetConfig), 'config not supported {}'.format(config)
    # config_data = vars(config)
    # log_params(config_data)


class LinkNetPredictor(object):
    """docstring for LinkNetPredictor."""

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
