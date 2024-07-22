"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import random
import requests
import math
from shapely.geometry import Polygon
import sys


def _matrix_iou(a,b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def my_iou(a,roi):
    cond0 = np.min(a[:,0]) > roi[0]
    cond1 = np.min(a[:,1]) > roi[1]
    cond2 = np.max(a[:,2]) < roi[2]
    cond3 = np.max(a[:,3]) < roi[3]
    return cond0 & cond1 & cond2 & cond3

def _drop_area(image, bounding_boxes, pct_drop=0.1):
    min_side = 32
    max_side = 128
    if image.shape[0] == 3:
        _, height, width = image.shape
    else:
        height, width, _ = image.shape
    plate_polygons = []

    for box in bounding_boxes:
        plate_polygons.append(Polygon([(x, y) for x, y in zip(box[0::2], box[1::2])]))

    im_area_drop = int((height * width) * pct_drop)

    dropped_areas = []
    im = image.copy()
    if width - max_side < 0 or height - max_side < 0:
        return image

    cnt = 2000
    while im_area_drop > 0 and cnt > 0:
        x0 = random.randint(0, width - max_side)
        y0 = random.randint(0, height - max_side)
        da_w = random.randint(min_side, max_side)
        da_h = random.randint(min_side, max_side)
        poly = Polygon([(x0, y0), (x0 + da_w, y0), (x0 + da_w, y0 + da_h), (x0, y0 + da_h)])
        overlaps = False
        for pl_poly in plate_polygons:
            if pl_poly.intersects(poly):
                overlaps = True
                break

        if not overlaps:
            dropped_areas.append([x0, y0, da_w, da_h])
            im_area_drop -= da_w * da_h
        cnt -= 1

    x = bounding_boxes[0, 0::2].astype(int)
    y = bounding_boxes[0, 1::2].astype(int)

    for qq, q in enumerate(dropped_areas):
        x0, y0, da_w, da_h = q
        prob_color = 0.5
        patch_h = da_h
        patch_w = da_w
        if random.random() >= prob_color:
            patch_h = 1
            patch_w = 1

        if im.shape[0] == 3:
            im[:, y0:y0+da_h, x0:x0+da_w] = np.random.rand(3, patch_h, patch_w) * 255
        else:
            im[y0:y0+da_h, x0:x0+da_w, :] = np.random.rand(patch_h, patch_w, 3) * 255

    return im

def _crop(image, boxes, labels, corners):
    """Perform random crop of image.

    :param np.ndarray image: With dimension ordering HWC.
    :param np.ndarray boxes: Of shape (1, 4) with values xmin, xmax, ymin, and ymax.
    :param labels: Class label (2 for license plate, 0 for background).
    :param np.ndarray corners: Of shape (1, 8) with non-rectangular corner points.
    :return tuple: Of image, boxes, labels, and corners.
    """
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (1.0, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels, corners

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))
            iou = _matrix_iou(boxes, roi[np.newaxis])
            iou_ = my_iou(boxes, roi)
            if not iou_:
                continue

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue
            # print('------------------------')
            # print("i: "+str(i)+" iou "+str(iou.item()))
            # print(iou)
            # print(roi)
            # print(boxes)
            # print('------------------------')
            # roi = np.array((311, 220, 562, 438))

            #   roi [x0 y0 x1 y1]
            #
            #  (x0,y0) --------------------+
            #     |                        |
            #     |                        |
            #     |                        |
            #     +---------------------(x1,y1)

            # print("iou "+str(iou))
            # do the actual crop
            image_t = image[roi[1]:roi[3], roi[0]:roi[2]].copy()

            # refine boxes_t
            # calculate center[x y]
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            # 
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            corners_t = corners[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2] # x0 y0
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2] # x1 y1

            # print("boxes t man")
            # print(boxes_t)
            # print(boxes_t[:, :2])
            # print(boxes_t[:, 2:])
            # print("---------")
            # print(corners_t)
            # print(corners_t[:, :2])
            # print(corners_t[:, 2:4])
            # print(corners_t[:, 4:6])
            # print(corners_t[:, 6:8])
            # print("ROI")
            # print(roi)
            
            #
            #       Corners coordinate system
            #       In general, this is not regular.
            #
            #       corners [x0 y0 x1 y1 x2 y2 x3 y3]
            #
            #       (x0,y0) -------------------- (x1, y1)
            #          |                            |
            #          |                            |
            #          |                            |
            #       (x2, y2) ------------------- (x2,y2)

            corners_o = corners[mask].copy()
            # x0 y0
            corners_t[:, :2] = np.maximum(corners_t[:, :2], roi[:2])
            corners_t[:, :2] -= roi[:2] # x0 y0
            # x1 y1
            corners_t[:, 2] = np.minimum(corners_t[:, 2], roi[2])
            corners_t[:, 3] = np.maximum(corners_t[:, 3], roi[1])
            corners_t[:, 2:4] -= roi[:2] # x1 y1
            # if corners_t[:, 2] < roi[2]:
                # t = corners_o[:, 3] * (roi[2]/corners_o[:, 2])
                # corners_t[:, 3] = int(t)
                
            # x2 y2
            corners_t[:, 4:6] = np.minimum(corners_t[:, 4:6], roi[2:])
            corners_t[:, 4:6] -= roi[:2] # x0 y0
            # x3 y3
            corners_t[:, 6] = np.maximum(corners_t[:, 6], roi[0])
            corners_t[:, 7] = np.minimum(corners_t[:, 7], roi[3])
            corners_t[:, 6:] -= roi[:2] # x0 y0
            

            # print("New corners_t")
            # print(corners_t)
            # c  = corners_t.copy()
            # for i in range(0,6,2):
            #     cv2.line(image_t, (c[:, i],c[:, i+1]),(c[:, i+2], c[:, i+3]),(0,0,255),2)
    
            # cv2.line(image_t, (c[:, 0],c[:, 1]),(c[:, 6], c[:, 7]),(0,0,255),2)
            

            # cv2.line(image_t, (c[:, 0],c[:, 1]), (c[:, 2], c[:, 3]),(0,0,255),2)
            # # cv2.line(image_t, (c[:, 2], c[:, 3]),(c[:, 4],c[:, 5]),(0,0,255),2)
            # cv2.imwrite("RPOI.png",image_t)
            # c = corners_o.copy()
            # cv2.imwrite("image.png",image)
            # cv2.line(image, (c[:, 0],c[:, 1]), (c[:, 2], c[:, 3]),(0,0,255),2)
            # cv2.imwrite("RPOI_b.png",image)

            
            # sys.exit() 

            # print("roi")
            # print(roi)
            return image_t, boxes_t,labels_t, corners_t
        return image, boxes, labels, corners


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    # Do alpha/beta augmentation in one operation
    operation = random.randrange(4)
    if operation == 0:
        _convert(image, alpha=random.uniform(0.6, 1.4))
    elif operation == 1:
        _convert(image, beta=random.uniform(-24, 24))
    elif operation == 2:
        _convert(image, alpha=random.uniform(0.6, 1.4), beta=random.uniform(-24, 24))
    elif operation == 3:
        pass

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # 25% chance
    if random.randrange(4) == 0:
        # Convert to grayscale!
        # TODO: should be a more efficient way to do this... (e.g., opencv decolor)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    return image


def _expand(image, boxes, corners, fill, p):
    height, width, depth = image.shape

    if random.random() > p:
        mask_image = np.empty(
            (height, width, depth),
            dtype=image.dtype)

        mask_image[:, :] = 0
        return image, boxes, corners, mask_image

    for _ in range(50):
        scale = random.uniform(1,4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        corners_t = corners.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)
        corners_t[:, 0:2] += (left, top)
        corners_t[:, 2:4] += (left, top)
        corners_t[:, 4:6] += (left, top)
        corners_t[:, 6:8] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)

        mask_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)

        mask_image[:, :] = 255
        mask_image[top:top + height, left:left + width] = 0

        # cv2.imshow("Mask", mask_image)

        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t, corners_t, mask_image

    return image, boxes, corners


def _preproc_resize(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_NEAREST]
    interp_method = interp_methods[random.randrange(len(interp_methods))]

    if isinstance(insize, int):
        dst = (insize, insize)
    else:
        dst = (insize[1], insize[0])

    image = cv2.resize(image, dst, interpolation=interp_method)

    return image


def _preproc_for_test(image, mean):
    image = image.astype(np.float32)


    image -= mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets_, path):
        """Apply augmentations to original image.

        :param np.ndarray image: With shape order HWC and BGR channels.
        :param np.ndarray targets_: Bounding box location, corners, and class.
        :return torch.tensor:
        """

        targets = targets_[:, 0:5]
        corners = targets_[:, 5:13].copy()
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # in case there are no bounding boxes..
        if boxes.sum() + corners.sum() == 0:
            targets = np.zeros(targets_.shape)
            image = _preproc_resize(image, self.resize)
            image = _preproc_for_test(image, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        corners_o = corners.copy()

        # normalize coordinates inside [0,1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        corners_o[:, 0::2] /= width_o
        corners_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o, corners_o))

        # Apply augmentation classes
        image_t, boxes, labels, corners = _crop(image, boxes, labels, corners)
        image_t, boxes, corners, mask_image = _expand(image_t, boxes, corners, 0, self.p)
        height, width, depth = image_t.shape
        image_t = _preproc_resize(image_t, self.resize)
        mask_image = _preproc_resize(mask_image, self.resize)
        resized_h, resized_w, resized_d = image_t.shape
        image_t = _distort(image_t)

        if random.uniform(0, 1) > 0.9:
            image_t = _drop_area(image_t, corners)

        # Apply the mean to the image
        mean_image = np.empty(
            (resized_h, resized_w, resized_d),
            dtype=image.dtype)
        mean_image[:, :] = self.means
        image_t = (mean_image & mask_image) | image_t
        image_t = _preproc_for_test(image_t, self.means)

        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        corners = corners.copy()
        corners[:, 0::2] /= width
        corners[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.

        # mask_b is either true or false
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        corners_t = corners[mask_b]

        if len(boxes_t) == 0:
            image = _preproc_resize(image_o, self.resize)
            image = _preproc_for_test(image, self.means)
            return torch.from_numpy(image), targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t, corners_t))  # [targets_[:,13]], [targets_[:,14]]))
        return torch.from_numpy(image_t), targets_t

class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[1]

        if isinstance(self.resize, int):
            dst = (self.resize, self.resize)
        else:
            dst = (self.resize[1], self.resize[0])


        img = cv2.resize(np.array(img), dst, interpolation = interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)




class BaseTransformAnnot(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, targets_, path):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[1]

        if isinstance(self.resize, int):
            dst = (self.resize, self.resize)
        else:
            dst = (self.resize[1], self.resize[0])


        # print(dst)
        # print(img.shape)
        width, height = dst
        height_o, width_o, _ = img.shape
        img = cv2.resize(np.array(img), dst, interpolation = interp_method).astype(np.float32)
        # cv2.imwrite('shata.png', img)
        targets = targets_.copy()
        targets[:, 0] = (targets[:, 0] / width_o)
        targets[:, 2] = (targets[:, 2] / width_o)
        targets[:, 1] = (targets[:, 1] / height_o)
        targets[:, 3] = (targets[:, 3] / height_o)
        # targets[:, 4]  i#s class
        targets[:, 5] = (targets[:, 5] / width_o)
        targets[:, 7] = (targets[:, 7] / width_o)
        targets[:, 9] = (targets[:, 9] / width_o)
        targets[:, 11] = (targets[:, 11] / width_o)

        targets[:, 6] = (targets[:, 6] / height_o)
        targets[:, 8] = (targets[:, 8] / height_o)
        targets[:, 10] = (targets[:, 10] / height_o)
        targets[:, 12] = (targets[:, 12] / height_o)

        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img), targets


class BaseTransformNearest(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = cv2.INTER_NEAREST #interp_methods[0]

        if isinstance(self.resize, int):
            dst = (self.resize, self.resize)
        else:
            dst = (self.resize[1], self.resize[0])


        img = cv2.resize(np.array(img), dst, interpolation = interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)
