import sys
import numpy as np
import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def match(threshold, truths, corners, priors, variances, labels, loc_t, conf_t, corn_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # Bipartite matching based on Jaccard index
    overlaps = jaccard(truths, point_form(priors))
    best_prior_overlap, best_prior_idx = overlaps.max(1)  # [1, num_objects] best prior for each ground truth
    best_truth_overlap, best_truth_idx = overlaps.max(0)  # [1, num_priors] best ground truth for each prior
    best_truth_overlap[best_prior_idx] = 2                # Ensure best prior

    # Ensure every ground truth matches with its prior of max overlap
    for truth_idx, prior_idx in enumerate(best_prior_idx):
        best_truth_idx[prior_idx] = truth_idx
    matched = truths[best_truth_idx]           # Shape: [num_priors, 4]
    matched_corn = corners[best_truth_idx]     # Shape: [num_priors, 8]
    conf = labels[best_truth_idx]              # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0   # Label as background
    
    # Encode offset of ground truths relative to matched priors
    loc = encode(matched, priors, variances)
    corn = encode_corners(matched_corn, priors, variances)

    # Write encoded data for the current sample index
    loc_t[idx] = loc    # [num_priors, 4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    corn_t[idx] = corn  # [num_priors, 8] encoded corners to learn


def encode_corners(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 8].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # xmin, ymin, xmax, ymax
    pf = point_form(priors)
    xmin = pf[:, 0].unsqueeze(1)
    ymin = pf[:, 1].unsqueeze(1)
    xmax = pf[:, 2].unsqueeze(1)
    ymax = pf[:, 3].unsqueeze(1)

    pf_8 = torch.cat((xmin, ymin,
                      xmax, ymin,
                      xmax, ymax,
                      xmin, ymax
                      ), 1).type_as(matched)

    t = (variances[0] * priors[:, 2:]).repeat(1, 4).type_as(matched)
    g_pt = (matched - pf_8).div_(t)
    return g_pt


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # Encode variance with distance between match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])

    # Match width/height to prior width/height
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # Target for smooth_l1_loss [num_priors,4]


def encode_multi(matched, priors, offsets, variances):
    
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] - offsets[:,:2]
    # encode variance
    #g_cxcy /= (variances[0] * priors[:, 2:])
    g_cxcy.div_(variances[0] * offsets[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode_corners(corn, priors, variances):
    """Adapted from https://github.com/Hakuyume/chainer-ssd

    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    # xmin, ymin, | xman, ymin, | xmax, ymax | xmin, ymax
    pf = point_form(priors)

    xmin = pf[:,0].unsqueeze(1)
    ymin = pf[:,1].unsqueeze(1)

    xmax = pf[:,2].unsqueeze(1)
    ymax = pf[:,3].unsqueeze(1)

    pf_8 = torch.cat((xmin, ymin, # xmin, ymin
                      xmax, ymin, # xmax, ymin
                      xmax, ymax, # xmax, ymax
                      xmin, ymax  # xmin, ymax
                      ), 1).type_as(corn)
    
    t = (variances[0] * priors[:,2:]).repeat(1,4).type_as(corn)
    cntr = (priors[:,:2]).repeat(1,4).type_as(corn)
    return  pf_8 + (corn).mul_(t)


def decode(loc, priors, variances):
    """Adapted from https://github.com/Hakuyume/chainer-ssd

    Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    # priors: [x0, y0, w0, h0]

    # return priors
    # print("decode")
    # print("priors.shape")
    # print(priors.shape)
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], # x0 + Dx*var(?)*w0, y0 + Dy*var(?)*h0
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1) # w0*exp(Dw*var(?)), h0*exp(Dh*var(?))
    
    # boxes: [x y w h] -> (w,y) of center
    # print("AFter concat")
    # print(boxes[0:5, :])
    # afaireis apo to (x,y) kentro to w,h /2 ara pas apo kentro panw aristera gwnia
    # meta pros8eteis sta w,h thn panw aristera gwnia kai pas sthn katw deksia
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_multi(loc, priors, offsets, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + offsets[:,:2]+ loc[:, :2] * variances[0] * offsets[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max