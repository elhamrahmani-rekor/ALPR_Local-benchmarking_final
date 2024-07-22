from base64 import b64encode
import os
import random
from statistics import mean
import string
import cv2
import numpy as np
from PIL import Image
import pdb
import pandas as pd
import torch
from torchvision import transforms,models
import pyiqa
from shapely.geometry import Polygon

class Img2Vec():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-18"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        # normalize the resized images as expected by resnet18
        # [0.485, 0.456, 0.406] --> normalized mean value of ImageNet, [0.229, 0.224, 0.225] std of ImageNet
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.inputDim = (224,224)
        self.cnn_transformation = transforms.Compose([transforms.Resize(self.inputDim)])

    def getVec(self, img):
        img2 = self.cnn_transformation(img)
        image = self.normalize(self.toTensor(img2)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)

        def copyData(m, i, o): embedding.copy_(o.data)

        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()

        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        cnnModel = models.resnet18(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512
        return cnnModel, layer

class IQA():
    def __init__(self):

        #these are the non-reference IQA models with lower values being = lower quality
        self.iqa_models = ['dbcnn', 'musiq', 'musiq-ava', 'musiq-koniq', 'musiq-paq2piq', 'musiq-spaq', 'nima','paq2piq']

        self.model_dict = {}
        for model in self.iqa_models:
            self.model_dict[model] = pyiqa.create_metric(model,device=torch.device('cuda'))
            #print(f"{model} - lower numbers better? {self.model_dict[model].lower_better}")
        pass

    def get_metric(self,img_tensor,model_name):
        score_nr = self.model_dict[model_name](img_tensor)
        return score_nr

    def run_iqa(self,img):
        output_dict = {}
        tensor_error=False
        try:
            img_tensor = pyiqa.img2tensor(img).unsqueeze(0)
        except:
            #print('tensor error - no image???')
            tensor_error=True

        for model in self.iqa_models:
            if not tensor_error:
                try:
                    result = self.get_metric(img_tensor,model).item()
                except:
                    result = None
            else:
                result=None

            output_dict[model] = result

        return output_dict



def focus_measure(img):
    """Get variance of laplacian of image for contrast calculation
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()


def hsv_values(img):
    """Get brightness/saturation of image scaled by image dimensions
    """
    hue_plate = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hue_plate)
    v_sumed = np.sum(v)
    value = v_sumed / (float(img.shape[0] * img.shape[1]))
    s_sumed = np.sum(s)
    saturation = s_sumed / (float(img.shape[0] * img.shape[1]))
    h_sumed = np.sum(h)
    hue = h_sumed / (float(img.shape[0] * img.shape[1]))
    output_dict = {}
    output_dict['hue'] = hue
    output_dict['value'] = value
    output_dict['saturation'] = saturation
    return output_dict

def black_masking(img,
                  coords):


    color = (0, 0, 0)

    thickness = -1

    for coord in coords:
        img = cv2.rectangle(img, coord[0], coord[1], color, thickness)
    return img


def image_color_mean(img):
    """Get mean of color channels
    """
    out_dict = {}
    image = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    # Calculate the mean of each channel
    channels = cv2.mean(image)
    # Swap blue and red values (making it RGB, not BGR)
    observation = np.array([(channels[2], channels[1], channels[0])])

    colord = ['red','green','blue']

    for i,ob in enumerate(observation[0]):
            out_dict['mean_{}'.format(colord[i])] = ob

    return out_dict

def quality_metadata(img):
    iqa = IQA()
    output_dict = iqa.run_iqa(img)
    return output_dict

def vector_metadata(img):
    out_dict = {}
    try:
        vectorizor = Img2Vec()
        img_vec = vectorizor.getVec(Image.fromarray(np.array(img)))
        vec_list = [float('{:f}'.format(item)) for item in img_vec]
        out_dict['image_vector'] = vec_list
        out_dict['vector_model'] = vectorizor.modelName
    except:
        out_dict['image_vector'] = []
        out_dict['vector_model'] = ''

    try:
        hist_vals = get_hist(img)
        for key,val in hist_vals.items():
            out_dict[key] = val
    except:
        out_dict['blue_color_hist'] = []
        out_dict['green_color_hist'] = []
        out_dict['red_color_hist'] = []

    return out_dict



def image_metadata(img):
    out_dict = {}

    try:
        hsv = hsv_values(img)
        for key,val in hsv.items():
            out_dict[key] = val
    except:
        out_dict['hue'] = ''
        out_dict['value'] = ''
        out_dict['saturation'] = ''

    try:

        out_dict['focus_measure'] = focus_measure(img)
    except:
        out_dict['focus_measure'] = ''

    try:
        color_means = image_color_mean(img)
        for key,val in color_means.items():
            out_dict[key] = val
    except:
        out_dict['mean_red'] = ''
        out_dict['mean_green'] = ''
        out_dict['mean_blue'] = ''


    return out_dict


def get_hist(img):
    color = ('blue', 'green', 'red')
    output_dict = {}
    for i, color in enumerate(color):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        output_dict['{}_color_hist'.format(color)] = [list(item)[0] for item in histogram]


    return output_dict


def full_image_cv(image):
    """Get area and plate crop for display in webpage.

    :param str image_path: Full filepath to local image.
    :param [int] plate_corners: List of corner coordinates.
    :return str: Base64 encoded image data.
    """
    img_in = np.fromstring(image,np.uint8)
    # Load image and perform crops
    img = cv2.imdecode(img_in,1)
    if img is None:
        raise RuntimeError('Image could not be read')

    return img

def full_image_web(image_path):
    """Get area and plate crop for display in webpage.

    :param str image_path: Full filepath to local image.
    :param [int] plate_corners: List of corner coordinates.
    :return str: Base64 encoded image data.
    """

    # Load image and perform crops
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError('Image could not be read at {}'.format(image_path))

    # Encode to base64 for web transmission
    img_b64 = b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')
    return 'data:image/jpeg;base64,{}'.format(img_b64)


def levenshtein(s1, s2):
    """Calculate Levenshtein distance between two strings.

    :param str s1: First string to compare
    :param str s2: Second string to compare
    :return int: Minimum number of single-character edits required to match strings
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def bbox_stats(plate_corners_gt, plate_coordinates):
    """Shared geometry logic for is_overlapping and covers_whole_plate"""
    if isinstance(plate_corners_gt, str):
        gt_points = []
        for p in plate_corners_gt.split():
            gt_points.append(int(p))

        p1_points = []
        for i in range(0, len(gt_points), 2):
            p1_points.append((gt_points[i], gt_points[i+1]))
    elif isinstance(plate_corners_gt, list) and \
            not isinstance(plate_corners_gt[0], dict):
        p1_points = []
        for x, y in zip(plate_corners_gt[0::2], plate_corners_gt[1::2]):
            p1_points.append((int(x), int(y)))

    else:
        p1_points = []
        for i in range(0, 4):
            gt = plate_corners_gt[i]
            p1_points.append((gt['x'], gt['y']))

    p2_points = []
    for i in range(0, 4):
        coords = plate_coordinates[i]
        p2_points.append((coords['x'], coords['y']))

    p1 = Polygon(p1_points)
    p2 = Polygon(p2_points)
    return {
        'gt_area': p1.area,
        'machine_area': p2.area,
        'intersection': p1.intersection(p2).area,
        'union': p1.union(p2).area}


def is_overlapping(gt, machine, thres=0.3):
    stats = bbox_stats(gt, machine)
    iou = stats['intersection'] / stats['union']
    return iou > thres


def covers_whole_plate(gt, machine, thres):
    stats = bbox_stats(gt, machine)
    return stats['intersection'] / stats['gt_area'] >= thres

def get_area_crop(image, corners, b64=True):
    """Crop image to area surrounding plate without perspective correction

    :param np.ndarray image: Pre-loaded OpenCV image
    :param [int(8)] corners: Plate coordinates
    :param bool b64: Whether to encode to a base64 string or not
    :return str or np.ndarray:
    """

    # Calculate plate size and determine appropriate crop
    assert len(corners) == 8, 'Corners must be four (x, y) pairs'
    img_height, img_width, channels = image.shape
    xs = sorted(corners[::2])
    ys = sorted(corners[1::2])
    plate_width = max(xs) - min(xs)
    plate_height = max(ys) - min(ys)
    box_dim = max([plate_width, plate_height])
    top = int(mean(ys[:2]) - (box_dim / 2))
    height = int(box_dim * 2)
    left = int(mean(xs[:2]) - (box_dim / 2))
    width = int(box_dim * 2)
    bottom = top + height
    right = left + width
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    if right > img_width:
        right = img_width
    if bottom > img_height:
        bottom = img_height
    crop_img = image[top:bottom, left:right]
    crop_height, crop_width, _ = crop_img.shape
    assert crop_width > 0 and crop_height > 0, 'Calculated crop has zero width/height'

    # Draw the corners around license plate
    for i in range(4):
        pt1 = (corners[2 * i] - left, corners[2 * i + 1] - top)
        pt2 = (corners[(2 * i + 2) % 8] - left, corners[(2 * i + 3) % 8] - top)
        cv2.line(crop_img, pt1, pt2, (255, 255, 0), 2)

    # Limit final image size
    if width > 400:
        crop_img = cv2.resize(crop_img, (300, 300))
    if b64:
        crop_b64 = b64encode(cv2.imencode('.jpg', crop_img)[1]).decode('utf-8')
        return 'data:image/jpeg;base64,' + crop_b64
    else:
        return crop_img


def get_images(image_path, plate_corners):
    """Get area and plate crop for display in webpage.

    :param str image_path: Full filepath to local image.
    :param [int] plate_corners: List of corner coordinates.
    :return 2-tuple(str): Base64 encoded image data.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError('Image could not be read at {}'.format(image_path))
    crop_b64 = get_perspective_corrected_crop(img, plate_corners, b64=True)
    area_b64 = get_area_crop(img, plate_corners, b64=True)
    return crop_b64, area_b64

def show_image_wait(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    # closing all open windows
    cv2.destroyAllWindows()
    return

def get_perspective_corrected_crop(image, corners, crop_width=250, crop_height=125, b64=True):
    """Crop image to just license plate and apply perspective correction

    :param str or np.ndarray image: Filepath or pre-loaded OpenCV image
    :param [int(8)] corners: Plate coordinates
    :param int crop_width: Crop pixel width
    :param int crop_height: Crop pixel height
    :param bool b64: Whether to encode to a base64 string or not
    :return str or np.ndarray:
    """
    if isinstance(image, str):
        img_path = image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f'Failed to load image at {img_path}')

    # Format corner list into coordinate pairs
    assert len(corners) == 8, 'Corners must be four (x, y) pairs'
    orig_coords = np.zeros((4, 2), dtype='float32')
    for i in range(0, 4):
        orig_coords[i][0] = corners[2*i]
        orig_coords[i][1] = corners[2*i + 1]

    # Apply perspective correction
    dst_coords = np.array([
        [0, 0],
        [crop_width - 1, 0],
        [crop_width - 1, crop_height - 1],
        [0, crop_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(orig_coords, dst_coords)
    warped_img = cv2.warpPerspective(image, M, (crop_width, crop_height))
    if b64:
        warped_b64 = b64encode(cv2.imencode('.jpg', warped_img)[1]).decode('utf-8')
        return 'data:image/jpeg;base64,' + warped_b64
    else:
        return warped_img




