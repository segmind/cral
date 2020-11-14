# from cral.tracking import log_params


class SSD300Config(object):
    """docstring for SSDConfig."""

    def __init__(
            self,
            aspect_ratios=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5],
                           [1.0, 2.0, 0.5]],
            strides=[8, 16, 32, 64, 100, 300],
            scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
            offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            max_boxes=300,
            variances=[0.1, 0.1, 0.2, 0.2],
            # score=0.3,
            alpha=1.0,
            neg_pos_ratio=3,
            pos_iou_threshold=0.5,
            neg_iou_limit=0.3):

        self.height = 300
        self.width = 300

        self.input_shape = (self.height, self.width, 3)

        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_limit = neg_iou_limit
        self.max_boxes_per_image = max_boxes

        self.aspect_ratios = aspect_ratios
        self.strides = strides

        self.variances = variances

        assert len(aspect_ratios) == len(offsets)
        self.offsets = offsets
        self.scales = scales
        self.max_boxes = max_boxes
        # needs to be an argument
        # parameters for loss function
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.two_boxes_for_ar1 = True
        self.clip_boxes = False
        self.normalize_coords = True
        self.input_anno_format = 'pascal_voc'

        self.coords = 'centroids'


def log_ssd_config_params(config):
    config_data = {}
    config_data['ssd_aspect_ratios'] = config.aspect_ratios
    config_data['ssd_strides'] = config.strides
    config_data['ssd_scales'] = config.scales
    config_data['ssd_offsets'] = config.offsets
    config_data['ssd_max_boxes'] = config.max_boxes
    config_data['ssd_variances'] = config.variances
    config_data['ssd_pos_iou_threshold'] = config.pos_iou_threshold
    config_data['ssd_neg_iou_limit'] = config.neg_iou_limit
    config_data['ssd_alpha'] = config.alpha
    config_data['ssd_neg_pos_ratio'] = config.neg_pos_ratio
    # return log_params(config_data)
