import cv2
import numpy as np
from cral.models.object_detection.YoloV3.base import yolo_eval
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

font = ImageFont.load_default()

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
    'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
    'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick',
    'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
    'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory',
    'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon',
    'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
    'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen',
    'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise',
    'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite',
    'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid',
    'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
    'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat',
    'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding."""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess_image(image_array, config):

    ih, iw, _c = image_array.shape
    h, w = config.input_shape

    # if not random:
    # resize image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = cv2.resize(image_array, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.ones(shape=(h, w, 3), dtype=np.uint8) * 128
    new_image[dy:dy + nh, dx:dx + nw, :] = image
    return new_image, scale, dy, dx


def annotate_image(image,
                   bboxes,
                   scores,
                   labels,
                   threshold=0.5,
                   label_dict=None):
    image = Image.open(image)
    Imagedraw = ImageDraw.Draw(image)
    # thickness = (image.size[0] + image.size[1]) // 300

    for box, label, score in zip(bboxes, labels, scores):
        if score < threshold:
            continue

        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(label, (left, top), (right, bottom))

        label_to_display = label
        if isinstance(label_dict, dict):
            label_to_display = label_dict[label]

        caption = '{}|{:.3f}'.format(label_to_display, score)

        colortofill = STANDARD_COLORS[label]
        Imagedraw.rectangle([left, top, right, bottom],
                            fill=None,
                            outline=colortofill,
                            width=3)

        display_str_heights = font.getsize(caption)[1]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        text_width, text_height = font.getsize(caption)
        margin = np.ceil(0.05 * text_height)
        Imagedraw.rectangle([(left, text_bottom - text_height - 2 * margin),
                             (left + text_width, text_bottom)],
                            fill=colortofill)

        Imagedraw.text((left + margin, text_bottom - text_height - margin),
                       caption,
                       fill='black',
                       font=font)

    return image


def freeze_model(model_path,
                 config,
                 num_classes,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5):
    # Load model, or construct model and load weights.
    yolo_model = keras.models.load_model(model_path, compile=False)

    boxes, scores, classes = yolo_eval(
        yolo_model.outputs,
        config.anchors,
        num_classes,
        config.input_shape,
        anchor_mask=config.anchor_mask,
        max_boxes=config.max_boxes,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold)

    prediction_model = keras.models.Model(yolo_model.input,
                                          [boxes, scores, classes])

    return prediction_model


def detect_image(model, image, config):

    boxed_image, scale, dy, dx = preprocess_image(np.array(image), config)

    image_data = boxed_image.astype(keras.backend.floatx())

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    boxes, scores, classes = model.predict(image_data)

    boxes[:, 0] = boxes[:, 0] - dy
    boxes[:, 2] = boxes[:, 2] - dy

    boxes[:, 1] = boxes[:, 1] - dx
    boxes[:, 3] = boxes[:, 3] - dx

    return boxes / scale, scores, classes
