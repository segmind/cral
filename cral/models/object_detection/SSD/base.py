import tensorflow as tf
from tensorflow import keras

from .models.keras_layer_DecodeDetections import DecodeDetections
from .models.keras_ssd300 import ssd_300

# from cral.tracking import log_params_decorator


# @log_params_decorator
def create_ssd_model(config,
                     num_classes,
                     feature_extractor='vgg16',
                     weights='imagenet'):
    assert feature_extractor == 'vgg16', 'only vgg16 supported for now'
    # K.clear_session()

    model, preprocess_input, predictor_sizes = ssd_300(
        weights=weights,
        image_size=config.input_shape,
        n_classes=num_classes,
        mode='training',
        l2_regularization=0.0005,
        scales=config.scales,
        aspect_ratios_per_layer=config.aspect_ratios,
        two_boxes_for_ar1=config.two_boxes_for_ar1,
        steps=config.strides,
        offsets=config.offsets,
        clip_boxes=config.clip_boxes,
        variances=config.variances,
        normalize_coords=config.normalize_coords,
        return_predictor_sizes=True)

    return model, preprocess_input, predictor_sizes


def decode_detections(training_model, config, **kwargs):
    decoded_predictions = DecodeDetections(
        # confidence_thresh=0.5,
        # iou_threshold=0.45,
        # top_k=200,
        # nms_max_output_size=400,
        coords=config.coords,
        normalize_coords=config.normalize_coords,
        img_height=config.height,
        img_width=config.width,
        name='SSD300',
        **kwargs)(
            training_model.output)

    model = keras.models.Model(
        inputs=training_model.input,
        outputs=[
            decoded_predictions[:, :, 2:], decoded_predictions[:, :, 1],
            tf.cast(decoded_predictions[:, :, 0], tf.int32)
        ])

    return model
