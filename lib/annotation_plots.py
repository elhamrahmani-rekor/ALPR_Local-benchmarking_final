import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pdb

def plot_timeseries(annot_frame, obj="P:0", roll_len=5):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(17, 10))
    ann_subframe = annot_frame[annot_frame.obj == obj]
    ann_subframe.index = list(np.arange(len(ann_subframe)))
    ax[0, 0].plot(ann_subframe["top"])
    ax[0, 0].plot(ann_subframe["top"].rolling(roll_len).mean())
    ax[0, 1].plot(ann_subframe["left"])
    ax[0, 1].plot(ann_subframe["left"].rolling(roll_len).mean())
    ax[1, 0].plot(ann_subframe["width"])
    ax[1, 0].plot(ann_subframe["width"].rolling(roll_len).mean())
    ax[1, 1].plot(ann_subframe["height"])
    ax[1, 1].plot(ann_subframe["height"].rolling(roll_len).mean())
    ax[0, 0].title.set_text("Top progression")
    ax[0, 1].title.set_text("Left progression")
    ax[1, 0].title.set_text("Width progression")
    ax[1, 1].title.set_text("Height progression")
    ax[0, 0].set_xlabel("Frame Number")
    ax[0, 0].set_ylabel("Box coordinate")
    ax[0, 1].set_xlabel("Frame Number")
    ax[0, 1].set_ylabel("Box coordinate")
    ax[1, 0].set_xlabel("Frame Number")
    ax[1, 0].set_ylabel("Box width")
    ax[1, 1].set_xlabel("Frame Number")
    ax[1, 1].set_ylabel("Box height")


def plot_deviations(annot_frame, figsize=(17, 10), obj="P:0", roll_len=5):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    ann_subframe = annot_frame[annot_frame.obj == obj]
    ann_subframe.index = list(np.arange(len(ann_subframe)))
    ax[0, 0].plot(ann_subframe["top"] - ann_subframe["top"].rolling(roll_len).mean())
    ax[0, 1].plot(ann_subframe["left"] - ann_subframe["left"].rolling(roll_len).mean())
    ax[1, 0].plot(ann_subframe["width"] - ann_subframe["width"].rolling(roll_len).mean())
    ax[1, 1].plot(ann_subframe["height"] - ann_subframe["height"].rolling(roll_len).mean())
    ax[0, 0].title.set_text("Top Roll Mean Deviation")
    ax[0, 1].title.set_text("Left Roll Mean Deviation")
    ax[1, 0].title.set_text("Width progression")
    ax[1, 1].title.set_text("Height Roll Mean Deviation")
    ax[0, 0].set_xlabel("Frame Number")
    ax[0, 0].set_ylabel("Coordinate deviation")
    ax[0, 1].set_xlabel("Frame Number")
    ax[0, 1].set_ylabel("Coordinate deviation")
    ax[1, 0].set_xlabel("Frame Number")
    ax[1, 0].set_ylabel("Width deviation")
    ax[1, 1].set_xlabel("Frame Number")
    ax[1, 1].set_ylabel("Height deviation")

class annot_methods():
    def __init__(self):
        pass


    def calc_box_size(self,annot_frame):
        lframe_len = max(annot_frame["frameid"])
        annot_frame.index = list(np.arange(len(annot_frame)))
        size_vec = list()
        for ind in annot_frame.index:
            area = annot_frame.loc[ind, 'height'] * annot_frame.loc[ind, 'width']
            size_vec.append(area)
        return np.array(size_vec)


    def calc_frame_int_over_union(self,annot_frame, i):
        i = int(i)
        annot_frame.loc[:,'frameid'] = annot_frame.loc[:,'frameid'].astype(int)
        lframe_len = max(annot_frame["frameid"])
        annot_frame.index = list(np.arange(len(annot_frame)))

        coord_vec = np.zeros((lframe_len + 1, 4))


        coord_vec[annot_frame["frameid"].values, 0] = annot_frame["left"]
        coord_vec[annot_frame["frameid"].values, 1] = annot_frame["top"]
        coord_vec[annot_frame["frameid"].values, 2] = annot_frame["width"]
        coord_vec[annot_frame["frameid"].values, 3] = annot_frame["height"]

        boxA = [
            coord_vec[i, 0],
            coord_vec[i, 1],
            coord_vec[i, 0] + coord_vec[i, 2],
            coord_vec[i, 1] + coord_vec[i, 3],
        ]
        try:
            boxB = [
                coord_vec[i + 1, 0],
                coord_vec[i + 1, 1],
                coord_vec[i + 1, 0] + coord_vec[i + 1, 2],
                coord_vec[i + 1, 1] + coord_vec[i + 1, 3],
            ]

            return self.bb_int_over_union(boxA, boxB)
        except IndexError:
            return 0

    def bb_int_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


    def create_annot_frame(self,annots):
        widths = []
        tops = []
        lefts = []
        heights = []
        objs = []
        files = []
        classids = []
        frameids = []

        objhashs = []

        roadtypes = []
        weathertypes=[]
        distancetypes=[]
        daynights=[]
        imagequalitys=[]

        classconfidences = []
        percvis = []
        occupancys = []

        for i, ann in enumerate(annots):
            for label in ann["annotations"]:
                frameids.append(i)

                if 'class_confidence_annotator' in label['label-category-attributes'].keys():
                    classconfidences.append(label['label-category-attributes']['class_confidence_annotator'])
                else:
                    classconfidences.append(np.NaN)

                if 'percent_visible' in label['label-category-attributes'].keys():
                    percvis.append(label['label-category-attributes']['percent_visible'])
                else:
                    percvis.append(np.NaN)

                if 'occupancy' in label['label-category-attributes'].keys():
                    occupancys.append(label['label-category-attributes']['occupancy'])
                else:
                    occupancys.append(np.NaN)

                files.append(ann["frame"])

                if 'road_type' in ann['frame-attributes'].keys():
                    roadtypes.append(ann['frame-attributes']['road_type'])
                else:
                    roadtypes.append(np.NaN)
                if 'weather_type' in ann['frame-attributes'].keys():
                    weathertypes.append(ann['frame-attributes']['weather_type'])
                else:
                    weathertypes.append(np.NaN)
                if 'road_distance' in ann['frame-attributes'].keys():
                    distancetypes.append(ann['frame-attributes']['road_distance'])
                else:
                    distancetypes.append(np.NaN)

                if 'day_night' in ann['frame-attributes'].keys():
                    daynights.append(ann['frame-attributes']['day_night'])
                else:
                    daynights.append(np.NaN)

                if 'image_quality' in ann['frame-attributes'].keys():
                    imagequalitys.append(ann['frame-attributes']['image_quality'])
                else:
                    imagequalitys.append(np.NaN)

                width = label["width"]
                widths.append(width)
                height = label["height"]
                heights.append(height)
                top = label["top"]
                tops.append(top)
                left = label["left"]
                lefts.append(left)
                obj = label["object-name"]
                objs.append(obj)
                classid = label["class-id"]
                classids.append(classid)
                objhashs.append(label['object-id'])

        annot_frame = pd.DataFrame()
        annot_frame["frameid"] = frameids
        annot_frame["file"] = files
        annot_frame["obj"] = objs
        annot_frame['objectid'] = objhashs
        annot_frame["left"] = lefts
        annot_frame["top"] = tops
        annot_frame["width"] = widths
        annot_frame["height"] = heights
        annot_frame['road_type'] = roadtypes
        annot_frame['weather_type'] = weathertypes
        annot_frame['road_distance'] = distancetypes
        annot_frame['day_night'] = daynights
        annot_frame['image_quality'] = imagequalitys
        annot_frame['class_confidence_annotator'] = classconfidences
        annot_frame['percent_visible'] = percvis
        annot_frame['occupancy'] = occupancys
        annot_frame = annot_frame.sort_values(by='file')
        return annot_frame



