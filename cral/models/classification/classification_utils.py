import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, GlobalAveragePooling2D)


def BatchNorm_dense(input_tensor, activation_fn, num_units, dropout):
    x = Dense(num_units, activation=None)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(activation_fn)(x)

    if dropout is not None:
        x = Dropout(dropout)(x)
    return x


def densely_connected_head(feature_extractor_model, fully_connected_layer,
                           dropout_rate, hidden_layer_Activation):
    """Adds densely connected head to backbone.

    Args:
        feature_extractor_model (keras.Model): the backbone model
        fully_connected_layer (list, tuple): A list or tuple indicating
                number of neurons per hidden layer
        dropout_rate (list, tuple): A list or tuple indicating the dropout
                rate per hidden layer
        hidden_layer_Activation (list, tuple): A list or tuple indicating the
                activation function per hidden layer

    Returns:
        tf.tensor: tensor of the final hidden layer to which output layer
                should be attached.
    """
    base_model_tensor = feature_extractor_model.output

    x = GlobalAveragePooling2D()(base_model_tensor)

    for index, unit in enumerate(fully_connected_layer):

        if isinstance(hidden_layer_Activation, (tuple, list)):
            activation_fn = hidden_layer_Activation[index]
        else:
            activation_fn = hidden_layer_Activation

        if dropout_rate is None:
            dropout = None
        elif isinstance(dropout_rate, (tuple, list)):
            dropout = dropout_rate[index]
        else:
            dropout = dropout_rate

        x = BatchNorm_dense(
            input_tensor=x,
            activation_fn=activation_fn,
            num_units=unit,
            dropout=dropout)

    return x


class MLPConfig:
    """Config for Multilayered Perceptron  at the top of the model.

    Attributes:
        dropout_rate (list, tuple): A list or tuple indicating the dropout
                rate per hidden layer
        final_layer_activation (str, optional): str indicating the activation
                function of the final prediction layer
        fully_connected_layer (list, tuple): A list or tuple indicating number
                of neurons per hidden layer
        height (int): height of images
        hidden_layer_activation (list, tuple): A list or tuple indicating the
                activation function per hidden layer
        width (int): width of images
    """

    def __init__(self,
                 height,
                 width,
                 fully_connected_layer=[],
                 dropout_rate=None,
                 hidden_layer_activation='relu',
                 final_layer_activation='softmax'):
        """Config for Multilayered Perceptron.

        Args:
            height (int): height of images
            width (int): width of images
            fully_connected_layer (list, tuple): A list or tuple indicating
                    number of neurons per hidden layer
            dropout_rate (list, tuple): A list or tuple indicating the dropout
                    rate per hidden layer
            hidden_layer_activation (list, tuple): A list or tuple indicating
                    the activation function per hidden layer
            final_layer_activation (str, optional): str indicating the
                    activation function of the final prediction layer
        """
        self.height = height
        self.width = width
        self.fully_connected_layer = fully_connected_layer
        self.dropout_rate = dropout_rate
        self.hidden_layer_activation = hidden_layer_activation
        self.final_layer_activation = final_layer_activation


class ClassificationPredictor(object):
    """docstring for ClassificationPredictor."""

    def __init__(self, model, preprocessing_func, size):
        self.model = model
        self.preprocessing_func = preprocessing_func
        self.size = size

    def load_image(self, image_path):
        return np.array(
            keras.preprocessing.image.load_img(path=image_path),
            dtype=np.uint8)

    # @abstractmethod
    def predict(self, image_path):
        image_array = self.load_image(image_path)
        image_array = tf.image.resize(image_array, self.size)
        preprocessed_image = self.preprocessing_func(image_array)
        preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)
        result = self.model.predict(preprocessed_image)
        return np.argmax(result, axis=-1), np.amax(result, axis=-1)
