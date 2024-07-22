import sys
sys.path.append('/app')
from itertools import product
from math import sqrt,ceil
import torch
from torch.autograd import Function
from lib.plate_detector.utils.box_utils import decode, decode_corners
from copy import deepcopy
from lib.plate_detector.utils.config import COCO_mobile_1080_2

def gen_priors(x):
    feature_maps = []
    for step in [16.0, 32.0, 64.0, 128.0, 256.0, 512.0]:
        fm_y = ceil(x.shape[2] / step)
        fm_x = ceil(x.shape[3] / step)
        feature_maps.append([fm_y, fm_x])
    prior_box = PriorBox(COCO_mobile_1080_2)
    with torch.no_grad():
        priors = torch.Tensor(prior_box.forward(feature_maps))
    return priors

class Detect(Function):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, cfg):
        self.num_classes = num_classes
        self.background_label = bkg_label

        self.variance = cfg['variance']

    def forward(self, predictions, prior):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """

        loc, conf, corn = predictions

        loc_data = loc.data
        corn_data = corn.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size

        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(1, self.num_priors, 4)
        self.corners = torch.zeros(1, self.num_priors, 8)
        self.scores = torch.zeros(1, self.num_priors, self.num_classes)

        if loc_data.is_cuda:
            self.boxes = self.boxes.cuda()
            self.scores = self.scores.cuda()
            self.corners = self.corners.cuda()

        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors, self.num_classes)
            self.boxes.expand_(num, self.num_priors, 4)
            self.scores.expand_(num, self.num_priors, self.num_classes)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_corners = decode_corners(corn_data[i], prior_data, self.variance)
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            self.boxes[i] = decoded_boxes
            self.corners[i] = decoded_corners
            self.scores[i] = conf_scores

        return self.boxes, self.corners, self.scores



class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg, debug=False):
        self.debug = debug
        if isinstance(cfg, dict):
            return self.__init__dict(cfg)
        else:
            print(type(cfg))
            print('not implemented yet!')
            sys.exit()

    def __init__dict(self, cfg):
        self.cfg = deepcopy(cfg)
        self.image_size_x = cfg['img_w']
        self.image_size_y = cfg['img_h']

        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        if not 'steps_x' in cfg:
            self.steps_x = cfg['steps']
            self.steps_y = cfg['steps']
        else:
            self.steps_x = cfg['steps_x']
            self.steps_y = cfg['steps_y']

        if not 'min_sizes_x' in cfg:
            self.min_sizes_x = cfg['min_sizes']
            self.min_sizes_y = cfg['min_sizes']
        else:
            self.min_sizes_x = cfg['min_sizes_x']
            self.min_sizes_y = cfg['min_sizes_y']

        if not 'max_sizes_x' in cfg:
            self.max_sizes_x = cfg['max_sizes']
            self.max_sizes_y = cfg['max_sizes']
        else:
            self.max_sizes_x = cfg['max_sizes_x']
            self.max_sizes_y = cfg['max_sizes_y']

        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.skip_tall = False  # cfg['skip_tall']
        self.skip_tallest = False  # cfg['skip_tallest']

        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def get_num_priors_per_layer(self):
        num_priors_per_layer = []
        for k, f in enumerate(self.feature_maps):
            n = 2
            # rest of aspect ratios
            m = max(self.aspect_ratios[k])
            for ar in self.aspect_ratios[k]:
                n += 1
                if self.skip_tall:
                    continue
                if self.skip_tallest and ar == m:
                    continue
                n += 1
            num_priors_per_layer.append(n)
        return num_priors_per_layer

    def forward(self, feature_maps=None, xshape=None):
        """Generate prior boxes for the current set of feature maps

        For variable naming conventions, see pg. 5 of the original SSD paper
        under in the section "Choosing scales and aspect ratios for default boxes"
        https://arxiv.org/pdf/1512.02325.pdf

        :param [[int]] feature_maps: Spatial dimensions of each feature map
            from the backbone in ``(x, y)`` or ``(square, )`` format
        :param xshape: Unused, but seems like it should take the place of the
            hard-coded ``self._image_size_*`` attributes below
        :return torch.Tensor output: Default prior/anchor boxes across all
            feature maps with shape ``(N, 4)`` in xywh format. ``N`` can be
            determined by multiplying the area of each feature map by the
            number of prior boxes per grid location (either 6 or 4 depending
            on the aspect ratios specified in ``data.config``)
        """
        if feature_maps is None:
            feature_maps = self.feature_maps

        mean = []
        self._steps_y = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        self._steps_x = [16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        self._image_size_y = 1024
        self._image_size_x = 1024

        min_ratios = [0.0375, 0.13125, 0.234375, 0.46875, 0.65, 0.80]
        max_ratios = [0.13125, 0.234375, 0.46875, 0.65, 0.80, 0.95]

        for k, f in enumerate(feature_maps):
            if isinstance(f, int):
                f = [f, f]

            iter_space = product(range(f[0]), range(f[1]))  # Each possible location in the feature map grid
            f_k_x = self._image_size_x / self._steps_x[k]
            f_k_y = self._image_size_y / self._steps_y[k]

            s_k_x = min_ratios[k]
            s_k_y = min_ratios[k] * 0.5625
            s_k_x_max = max_ratios[k]
            s_k_y_max = max_ratios[k] * 0.5625
            s_k_x_max = sqrt(s_k_x * s_k_x_max)
            s_k_y_max = sqrt(s_k_y * s_k_y_max)
            for i, j in iter_space:
                cx = (j + 0.5) / f_k_x
                cy = (i + 0.5) / f_k_y

                # aspect_ratio: 1
                mean += [cx, cy, s_k_x, s_k_y]
                mean += [cx, cy, s_k_x_max, s_k_y_max]

                # Rest of aspect ratios
                # Skip tallest was an experimental feature since plates are usually wide aspect ratio
                m = max(self.aspect_ratios[k])
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k_x * sqrt(ar), s_k_y / sqrt(ar)]
                    if self.skip_tall:
                        continue
                    if self.skip_tallest and ar == m:
                        continue
                    mean += [cx, cy, s_k_x / sqrt(ar), s_k_y * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output[:, 0:2].clamp_(min=0)
            output[:, 2:4].clamp_(min=0, max=1)

        return output
