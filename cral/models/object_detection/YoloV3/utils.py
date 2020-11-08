"""Miscellaneous utility functions."""
from functools import reduce

import numpy as np
# from cral.tracking import log_params
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


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


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line,
                    input_shape,
                    random=True,
                    max_boxes=20,
                    jitter=.3,
                    hue=.1,
                    sat=1.5,
                    val=1.5,
                    proc_img=True):
    """random preprocessing for real-time data augmentation."""
    line = annotation_line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = input_shape
    box = np.array(
        [np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image_data = 0
        if proc_img:
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(
        1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data


class YoloV3Config(object):
    """docstring for YoloV3Config."""

    def __init__(self,
                 anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                          (59, 119), (116, 90), (156, 198), (373, 326)],
                 anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 max_boxes=20,
                 height=416,
                 width=416,
                 score=0.3,
                 iou=0.45):
        assert height % 32 == 0
        assert width % 32 == 0

        assert isinstance(anchors, list)

        self.anchors = np.array(anchors, dtype='float32').reshape(-1, 2)

        assert self.anchors.shape[0] % 3 == 0

        self.height = height
        self.width = width

        self.input_shape = (self.height, self.width)

        self.anchor_mask = anchor_mask
        self.max_boxes = max_boxes

        self.num_layers = len(self.anchors) // 3  # default setting
        self.input_anno_format = 'pascal_voc'

        # have tomake it dynamic
        assert self.num_layers == 3, 'currently support only 3 pyramid layers'

    def check_equality(self, other):
        if isinstance(other, YoloV3Config):
            old_dict = other.__dict__
            curr_dict = self.__dict__
            for (key1, val1), (key2, val2) in zip(old_dict.items(),
                                                  curr_dict.items()):
                if not ((key1 == key2) and str(val1) == str(val2)):
                    if key1 == key2:
                        assert False, f'Value of {key1} changes from {val1} --> {val2}'  # noqa: E501
                    else:
                        assert False, f'Value of ({key1} : {val1}) changes to ({key2} : {val2})'  # noqa: E501
                    return False
            return True
        return False


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes,
                          anchor_mask):
    """Preprocess true boxes to training input format.

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    """
    assert (true_boxes[..., 4] <
            num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = anchor_mask

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [
        input_shape // {
            0: 32,
            1: 16,
            2: 8
        }[l] for l in range(num_layers)  # noqa: E741
    ]
    y_true = [
        np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(
            anchor_mask[l]), 5 + num_classes),
                 dtype='float32') for l in range(num_layers)  # noqa: E741
    ]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):  # noqa: E741
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] *
                                 grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] *
                                 grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def data_generator(annotation_path, batch_size, input_shape, anchors,
                   num_classes, anchor_mask):
    """data generator for fit_generator."""
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(
                annotation_lines[i], input_shape, random=False)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors,
                                       num_classes, anchor_mask)
        yield [image_data, *y_true], np.zeros(batch_size)


def log_yolo_config_params(config):
    config_data = {}
    config_data['yolo_height'] = config.height
    config_data['yolo_width'] = config.width
    config_data['yolo_anchors'] = config.anchors,
    config_data['yolo_anchor_mask'] = config.anchor_mask
    config_data['yolo_max_boxes'] = config.max_boxes
