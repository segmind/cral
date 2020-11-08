from functools import partial

from cral.models.object_detection.object_detection_utils import (
    Predictor, annotate_image, convert_to_coco)
from cral.models.object_detection.retinanet import (RetinanetConfig,
                                                    RetinanetGenerator,
                                                    get_retinanet,
                                                    get_retinanet_fromconfig)
from cral.models.object_detection.YoloV3 import YoloV3Config

retinanet_resnet50 = partial(get_retinanet, 'resnet50')
retinanet_resnet101 = partial(get_retinanet, 'resnet101')
retinanet_resnet152 = partial(get_retinanet, 'resnet152')
retinanet_resnet50v2 = partial(get_retinanet, 'resnet50v2')
retinanet_resnet101v2 = partial(get_retinanet, 'resnet101v2')
retinanet_resnet152v2 = partial(get_retinanet, 'resnet152v2')
retinanet_densenet121 = partial(get_retinanet, 'densenet121')
retinanet_densenet169 = partial(get_retinanet, 'densenet169')
retinanet_densenet201 = partial(get_retinanet, 'densenet201')
retinanet_mobilenet = partial(get_retinanet, 'mobilenet')
retinanet_mobilenetv2 = partial(get_retinanet, 'mobilenetv2')  #guessed
retinanet_vgg16 = partial(get_retinanet, 'vgg16')
retinanet_vgg19 = partial(get_retinanet, 'vgg19')
retinanet_efficientnetb0 = partial(get_retinanet, 'efficientnetb0')
retinanet_efficientnetb1 = partial(get_retinanet, 'efficientnetb1')
retinanet_efficientnetb2 = partial(get_retinanet, 'efficientnetb2')
retinanet_efficientnetb3 = partial(get_retinanet, 'efficientnetb3')
retinanet_efficientnetb4 = partial(get_retinanet, 'efficientnetb4')
retinanet_efficientnetb5 = partial(get_retinanet, 'efficientnetb5')
retinanet_efficientnetb6 = partial(get_retinanet, 'efficientnetb6')
retinanet_xception = partial(get_retinanet, 'xception')  #guessed
retinanet_detnet = partial(get_retinanet, 'detnet')
