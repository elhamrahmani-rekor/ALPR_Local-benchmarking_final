import sys
sys.path.append('/app')
import json
import io
import pdb
import pandas as pd
from lib.cloud_data_manager import s3_manager
from tqdm import tqdm
import onnxruntime
import cv2
import numpy as np
import os

class classifier_inference():
    def __init__(self,
                 model_path='/app/data/fhwa_classifier/sts-classifier.onnx',
                 json_path='/app/data/fhwa_classifier/sts-classifier.json',
                 gpu_avail=False,
                 crop_size=(64, 32)
                 ):

        if gpu_avail:
            self.session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        #load json file with class strings for matching
        with open(json_path,"r") as read_content:
            self.class_mapper = json.load(read_content)

        #Use json to format strings
        self.class_mapper['fhwa_class'] = {int(cls_idx): cls_name for cls_idx, cls_name in self.class_mapper['fhwa_class'].items()}
        self.class_mapper['vehicle'] = {int(cls_idx): cls_name for cls_idx, cls_name in self.class_mapper['vehicle'].items()}
        self.output_names = ['fhwa_class', 'vehicle']

        #normalize
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        # self.input_shape = (128, 64) #w,h
        self.input_shape = crop_size #w,h
        self.s3m = s3_manager()
        pass

    def run_inference(self,img):
        output_dict = {}

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_shape)
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.transpose(img, axes=[0, 3, 1, 2])

            # Run inference
            # get inputs
            ort_inputs = {self.session.get_inputs()[0].name: img}
            # get outputs from inputs
            ort_outputs = self.session.run(None, ort_inputs)

            # split outputs
            fhwa_class_ort_output = ort_outputs[0]
            vehicle_ort_output = ort_outputs[1]

            # store outputs
            output_dict['vehicle'] = {}
            for i in range(len(vehicle_ort_output[0])):
                output_dict['vehicle'][self.class_mapper['vehicle'][i]] = vehicle_ort_output[0][i]

            output_dict['fhwa_class'] = {}
            for i in range(len(fhwa_class_ort_output[0])):
                output_dict['fhwa_class'][self.class_mapper['fhwa_class'][i]] = fhwa_class_ort_output[0][i]

            # create best prediction
            best_predictions = {k: v for k, v in sorted(output_dict['fhwa_class'].items(), key=lambda item: item[1],
                                       reverse=True)}

            best_pred = max(best_predictions, key=best_predictions.get)

            #final_pred = self.class_mapper['fhwa_class'][best_pred]

            output_dict['prediction'] ={}
            output_dict['prediction']['fhwa_class_label'] = best_pred
            output_dict['prediction']['fhwa_class_confidence'] = best_predictions[best_pred]
            return output_dict
        except:
            return output_dict
