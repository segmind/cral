import cv2
import numpy as np
from cral.models.object_detection import Predictor
from tensorflow import keras
from tensorflow.keras.applications.resnet import \
    preprocess_input as resnet_preprocess_input  # noqa: F401


def load_model(checkpoint_path):

    pred_model = keras.models.load_model(
        filepath=checkpoint_path, compile=False)

    print('Weights are Loaded ...')

    return pred_model


def pad_resize(image, height, width, scale):
    """Summary.

    Args:
        image (TYPE): Description
        height (TYPE): Description
        width (TYPE): Description
        scale (TYPE): Description

    Returns:
        numpy nd.array: Description
    """
    # pad image
    padded_image = np.zeros(
        shape=(int(height), int(width), 3), dtype=image.dtype)
    h, w, _ = image.shape
    padded_image[:h, :w, :] = image

    # resize image
    resized_image = cv2.resize(
        padded_image, None, fx=scale, fy=scale).astype(keras.backend.floatx())
    return resized_image


def predict(model, image_path, preprocess_fn, min_side=800, max_side=1333):

    im = np.array(keras.preprocessing.image.load_img(path=image_path))

    smallest_side = min(im.shape[0], im.shape[1])
    largest_side = max(im.shape[0], im.shape[1])

    scale = min_side / smallest_side

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    images_batch = [
        cv2.resize(im, None, fx=scale, fy=scale).astype(keras.backend.floatx())
    ]

    images_batch = preprocess_fn(np.array(images_batch))

    bboxes, confidence, label = model.predict(images_batch)

    return bboxes[0].astype(int) / scale, confidence[0], label[0]


class RetinanetPredictor(Predictor):
    """docstring for RetinanetPredictor."""

    def __init__(self, min_side, max_side, *args, **kwargs):
        super(RetinanetPredictor, self).__init__(*args, **kwargs)
        self.min_side = min_side
        self.max_side = max_side

    def predict(self, image):
        im = self.load_image(image)

        smallest_side = min(im.shape[0], im.shape[1])
        largest_side = max(im.shape[0], im.shape[1])

        scale = self.min_side / smallest_side

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        image = cv2.resize(
            im, None, fx=scale, fy=scale).astype(keras.backend.floatx())

        images_batch = self.preprocessing_func(image)

        bboxes, confidence, label = self.model.predict(images_batch)

        return bboxes[0].astype(int) / scale, confidence[0], label[0]


if __name__ == '__main__':
    load_model('./checkpoints/prediction')
