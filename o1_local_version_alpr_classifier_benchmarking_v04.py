import sys
# sys.path.append('/app')
from lib.image_processing import image_metadata
from lib.cloud_data_manager import s3_manager
# from conf.app_config import app_settings
# from conf.log import log_
import os
# from scripts.download_models import retrieve_models
import argparse
from lib.utils import message_slack
# *************************
import json
import cv2
import pandas as pd
import onnxruntime
import numpy as np
import pdb
import torch
# from lib.plate_detector.utils.config import COCO_mobile_1080_2
# from lib.plate_detector.utils.prior_box import PriorBox,Detect,gen_priors
import math
import torch.nn.functional as F
from torchvision.ops import nms
import csv
from tqdm import tqdm
from io import StringIO, BytesIO
import alpr_benchmarking_constants
import time
import o2_alpr_daynight_classification_v02 as alpr_daynight_classification
import o3_alpr_triton_day_night_results


class BadDataLogger:
    def __init__(self):
        self.bad_data = []

    def add_bad_data(self, data):
        self.bad_data.append(data)


class AlprModels():
    def __init__(self):
        self.gpu_provider = True
        # self.s3m = s3_manager()
        self.img = None
        self.is_img_day = False
        self.bad_data_logger = BadDataLogger()
        self.image_embedding_collector = []
        self.region_code = {
            "asia": 0,
            "australia": 1,
            "eastern_europe": 2,
            "europe": 3,
            "india": 4,
            "middleeast": 5,
            "north_america": 6,
            "south_america": 7
        }

        self.country2region = {
            'ae': 'middleeast',
            'am': 'eastern_europe',
            'ar': 'south_america',
            'au': 'australia',
            'az': 'eastern_europe',
            'bh': 'middleeast',
            'br': 'south_america',
            'ca': 'north_america',
            'cn': 'asia',
            'co': 'south_america',
            'ec': 'south_america',
            'eg': 'middleeast',
            'eu': 'europe',
            'gb': 'europe',
            'ge': 'eastern_europe',
            'hk': 'asia',
            'id': 'asia',
            'il': 'middleeast',
            'in': 'india',
            'iq': 'middleeast',
            'ir': 'middleeast',
            'jo': 'middleeast',
            'jp': 'asia',
            'kg': 'eastern_europe',
            'kr': 'asia',
            'kw': 'middleeast',
            'kz': 'eastern_europe',
            'lb': 'middleeast',
            'mx': 'north_america',
            'my': 'asia',
            'nz': 'australia',
            'om': 'middleeast',
            'pk': 'middleeast',
            'py': 'south_america',
            'qa': 'middleeast',
            'rs': 'eastern_europe',
            'ru': 'eastern_europe',
            'sa': 'middleeast',
            'sg': 'asia',
            'th': 'asia',
            'tt': 'south_america',
            'us': 'north_america',
            'uz': 'eastern_europe',
            'ye': 'middleeast',
            'za': 'australia',
        }

    def get_region_code(self, country_code):
        region_value = self.country2region.get(country_code.lower())
        return self.region_code[region_value]


    def metadata_transformation(self, df):
        df_copy = df.copy()
        print("Original 'year' values:", df_copy['year'].unique())

        # ***********************transform year values into the working format*************************
        # Convert 'year' from float to int, handling NaNs and empty values
        def convert_year(value):
            if pd.isna(value) or value == '':
                return ''
            try:
                return int(float(value))  # Convert to float first to remove decimal and then to int
            except ValueError:
                return ''

        df_copy['year'] = df_copy['year'].apply(convert_year)
        df_copy['year'] = df_copy['year'].astype(
            str)  # Convert the whole column to string to display empty strings instead of <NA>
        print("reformatted 'year' values:", df_copy['year'])

        # ***********************create make_model filed in the dataframe*******************************
        # Create a new column 'initial_make' with the same values as 'make'. We may need the initial value of 'make' for future tracking.
        df_copy['initial_make'] = df_copy['make']

        # Replace NaN with an empty string for 'make' and 'model' before concatenation
        df_copy['make'] = df_copy['make'].fillna('')
        df_copy['model'] = df_copy['model'].fillna('')
        # Create 'make_model' by concatenating 'make' and 'model' with an underscore
        # Only concatenate if both 'make' and 'model' are not empty
        df_copy['make_model'] = df_copy.apply(
            lambda x: x['make'] + '_' + x['model'] if x['make'] and x['model'] else '', axis=1)
        print('concatenated make_model:', df_copy['make_model'])

        # *******************convert orientation values from float to string*******************************
        """
        this conversion is because orientation values are trained like this for example:
                           orientation": {"0": "0.0", "1": "135.0", "2": "180.0", "3": "225.0", 
                                          "4": "270.0", "5": "315.0", "6": "45.0", "7": "90.0"}
        """

        df_copy['orientation'] = df_copy['orientation'].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
        print("Reformatted 'orientation' values to integer string format.")

        # ***********************transform make values into the default-working format*************************
        def transform_make(value):
            abbreviation_make_values = ['ARO', 'Aro', 'aro',
                                        'AMC', 'Amc', 'amc',
                                        'BAW', 'Baw', 'baw',
                                        'BMC', 'Bmc', 'bmc',
                                        'BMW', 'Bmw', 'bmw',
                                        'CAM', 'Cam', 'cam',
                                        'CMC', 'Cmc', 'cmc',
                                        'DMC', 'Dmc', 'dmc',
                                        'FAW', 'Faw', 'faw',
                                        'GAZ', 'Gaz', 'gaz',
                                        'GMC', 'Gmc', 'gmc',
                                        'JAC', 'Jac', 'jac',
                                        'JMC', 'Jmc', 'jmc',
                                        'KTM', 'Ktm', 'ktm',
                                        'LDV', 'Ldv', 'ldv',
                                        'MG', 'Mg', 'mg',
                                        'TVR', 'Tvr', 'tvr',
                                        'UAZ', 'Uaz', 'uaz']

            if value in abbreviation_make_values:
                return value.upper()

            if pd.isna(value) or value == '':
                return ''

            # If there's a hyphen in the value, replace it with a space and capitalize each word
            if '-' in value:
                return ' '.join(word.capitalize() for word in value.split('-'))

            # If the value is a single word, check the first letter and change it if needed
            if ' ' not in value:
                return value if value[0].isupper() else value.capitalize()

            # If the value is multiple words separated by spaces, return it as is
            return value

        df_copy['make'] = df_copy['make'].apply(transform_make)
        print('make values:', df_copy['make'])
        df_copy.to_csv('/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_metadata/transformed_datasets.csv')
        return df_copy

    def get_metadata_csv(self, path_to_metadata):
        # df = self.s3m.get_bucket_object_df(bucket=bucket, key=path)
        df = pd.read_csv(path_to_metadata)
        self.test_data = self.metadata_transformation(df)
        pass

    def cleaning_model_values(self, cls_labels):
        for key, value in cls_labels['bodytype'].items():
            if value in alpr_benchmarking_constants.body_type_dict:
                cls_labels['bodytype'][key] = alpr_benchmarking_constants.body_type_dict[value]

        for key, value in cls_labels['make'].items():
            if value in alpr_benchmarking_constants.vehicle_make_dict:
                cls_labels['make'][key] = alpr_benchmarking_constants.vehicle_make_dict[value]

        # Adjust 'vehicle' values if needed
        for key, value in cls_labels['vehicle'].items():
            if value == "True":
                cls_labels['vehicle'][key] = "yes"

        # Convert orientation values from float strings to integer strings
        for key, value in cls_labels['orientation'].items():
            # Convert string with potential decimal to integer string
            cls_labels['orientation'][key] = str(int(float(value)))

        print(cls_labels)
        return cls_labels

    def convert_cls_labels_to_df(self, cls_labels):
        # Create a DataFrame with each key in cls_labels as a column header
        # The assumption is that each value in cls_labels is a list
        cls_labels_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in cls_labels.items()]))

        # Save DataFrame to CSV file
        cls_labels_csv_path = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_metadata/cls_labels.csv'  # specify your path here
        cls_labels_df.to_csv(cls_labels_csv_path, index=False)

    def init_vehicle_classifier(self, model_path, json_path, ):
        if self.gpu_provider:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        else:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # input_details = self.session.get_inputs()
        # for input in input_details:
        #     print("Input Name:", input.name)
        #     print("Input Type:", input.type)
        #     print("Input Shape:", input.shape)

        # load json file with class strings for matching
        with open(json_path, "r") as read_content:
            cls_labels = json.load(read_content)

        # Convert cls_labels dictionary to DataFrame
        self.convert_cls_labels_to_df(cls_labels)

        # cls_labels.
        # These 2 following lines should be removed when the issue of missing_plate and vehicle is solved
        # if 'missing_plate' in cls_labels:
        #     cls_labels.pop('missing_plate')
        # if 'vehicle' in cls_labels:
        #     cls_labels.pop('vehicle')

        self.cls_labels = self.cleaning_model_values(cls_labels)
        self.classifier_output_heads = ['bodytype', 'color', 'make', 'make_model', 'orientation', 'year', 'missing_plate', 'vehicle']
        self.model_dims = (224, 224)

        # Dictionaries to store daytime TP, FP, TN, FN for each head
        self.tp_dict_daytime = {head: 0 for head in self.classifier_output_heads}
        self.fp_dict_daytime = {head: 0 for head in self.classifier_output_heads}
        self.tn_dict_daytime = {head: 0 for head in self.classifier_output_heads}
        self.fn_dict_daytime = {head: 0 for head in self.classifier_output_heads}

        # Dictionaries to store nighttime TP, FP, TN, FN for each head
        self.tp_dict_nighttime = {head: 0 for head in self.classifier_output_heads}
        self.fp_dict_nighttime = {head: 0 for head in self.classifier_output_heads}
        self.tn_dict_nighttime = {head: 0 for head in self.classifier_output_heads}
        self.fn_dict_nighttime = {head: 0 for head in self.classifier_output_heads}

        # To store predicted and ground truth labels for each head for day and night separately
        self.predicted_labels_dict_daytime = {head: [] for head in self.classifier_output_heads}
        self.predicted_labels_dict_nighttime = {head: [] for head in self.classifier_output_heads}
        self.ground_truth_labels_dict_daytime = {head: [] for head in self.classifier_output_heads}
        self.ground_truth_labels_dict_nighttime = {head: [] for head in self.classifier_output_heads}

        # To store total accuracy
        self.accuracy_tracker_daytime = {head: {'correct_predictions': {}, 'total_count': {}} for head in
                                         self.classifier_output_heads}
        self.accuracy_tracker_nighttime = {head: {'correct_predictions': {}, 'total_count': {}} for head in
                                           self.classifier_output_heads}

        """ self.benchmark_results is supposed to store all final results 
           including raw_data, accuracy_metrics, confusion matrixes for all the heads and detailed_accuracy of the values of each head"""
        self.benchmark_results = {}
        self.s3m = s3_manager()
        pass

    def read_image(self, image_file):
        img = cv2.imread(image_file)
        self.img = img
        return img
        # pass

    def crop_vehicle(self, image_file, vehicle_id, image, bbox_int_list, target_dims=None):

        """Extract bounding box ROI

        :return: Unnormalized RGB pixels with shape ``(H, W, C)``
        """
        left, top, vehicle_width, vehicle_height = bbox_int_list
        right = left + vehicle_width
        bottom = top + vehicle_height
        top = max([0, top])
        left = max([0, left])
        bottom = min([bottom, image.shape[0]])
        right = min([right, image.shape[1]])

        crop = image[int(top):int(bottom), int(left):int(right)]

        if len(crop) == 0:  # It means it is a problematic data
            self.bad_data_logger.add_bad_data({
                'image_file': image_file,
                'vehicle_id': vehicle_id,
                'bbox': bbox_int_list,
                'log_description': 'The cropped image is returning an empty list'
            })
        if target_dims:
            resized = cv2.resize(src=crop, dsize=target_dims, interpolation=cv2.INTER_AREA)
            return resized
        else:
            return crop

    def update_confusion_matrix_daytime(self, prediction, ground_truth, head, predicted_label):
        if head in self.accuracy_tracker_daytime:
            if not pd.isna(ground_truth):  # Check if ground_truth is not null
                # if ground_truth is not None:  # Check if ground_truth is not null
                if ground_truth not in self.accuracy_tracker_daytime[head]['correct_predictions']:
                    self.accuracy_tracker_daytime[head]['correct_predictions'][ground_truth] = 0
                if ground_truth not in self.accuracy_tracker_daytime[head]['total_count']:
                    self.accuracy_tracker_daytime[head]['total_count'][ground_truth] = 0

                if prediction == ground_truth:
                    self.accuracy_tracker_daytime[head]['correct_predictions'][ground_truth] += 1
                self.accuracy_tracker_daytime[head]['total_count'][ground_truth] += 1

        if not pd.isna(ground_truth):
            if prediction == ground_truth:
                if prediction != 'unknown':
                    self.tp_dict_daytime[head] += 1
                else:
                    self.tn_dict_daytime[head] += 1
            else:
                if prediction != 'unknown':
                    self.fp_dict_daytime[head] += 1
                else:
                    self.fn_dict_daytime[head] += 1

        # Append predicted and ground truth labels for each head
        if not pd.isna(ground_truth):
            self.predicted_labels_dict_daytime[head].append(predicted_label)
            self.ground_truth_labels_dict_daytime[head].append(ground_truth)

    def update_confusion_matrix_nighttime(self, prediction, ground_truth, head, predicted_label):
        if head in self.accuracy_tracker_nighttime:
            if not pd.isna(ground_truth):  # Check if ground_truth is not null
                # if ground_truth is not None:  # Check if ground_truth is not null
                if ground_truth not in self.accuracy_tracker_nighttime[head]['correct_predictions']:
                    self.accuracy_tracker_nighttime[head]['correct_predictions'][ground_truth] = 0
                if ground_truth not in self.accuracy_tracker_nighttime[head]['total_count']:
                    self.accuracy_tracker_nighttime[head]['total_count'][ground_truth] = 0

                if prediction == ground_truth:
                    self.accuracy_tracker_nighttime[head]['correct_predictions'][ground_truth] += 1
                self.accuracy_tracker_nighttime[head]['total_count'][ground_truth] += 1

        if not pd.isna(ground_truth):
            if prediction == ground_truth:
                if prediction != 'unknown':
                    self.tp_dict_nighttime[head] += 1
                else:
                    self.tn_dict_nighttime[head] += 1
            else:
                if prediction != 'unknown':
                    self.fp_dict_nighttime[head] += 1
                else:
                    self.fn_dict_nighttime[head] += 1

        # Append predicted and ground truth labels for each head
        if not pd.isna(ground_truth):
            self.predicted_labels_dict_nighttime[head].append(predicted_label)
            self.ground_truth_labels_dict_nighttime[head].append(ground_truth)

    def update_raw_dataset(self, output_head, best_pred, confidence_score, test_dataset_row_index, inference_time):
        prediction_field_name = 'predicted_' + str(output_head)
        confidence_score_field_name = str(output_head) + '_confidence_score'

        self.test_data.loc[test_dataset_row_index, "inference_time(seconds)"] = inference_time
        self.test_data.loc[test_dataset_row_index, "is_day"] = self.is_img_day
        self.test_data.loc[test_dataset_row_index, prediction_field_name] = best_pred
        self.test_data.loc[test_dataset_row_index, confidence_score_field_name] = confidence_score

        # Update confusion matrix for each head
        successful_year_predication = False
        if output_head == 'vehicle':
            gt_label = 'yes'
        else:
            gt_label = self.test_data.loc[test_dataset_row_index, output_head]
            if output_head == 'year':
                start_year, end_year = map(int, best_pred.split('-'))
                year_list = [str(year) for year in range(start_year, end_year + 1)]
                if str(gt_label) in year_list:
                    successful_year_predication = True

        if successful_year_predication:
            if self.is_img_day:
                self.update_confusion_matrix_daytime(gt_label, gt_label, output_head, best_pred)
            else:
                self.update_confusion_matrix_nighttime(gt_label, gt_label, output_head, best_pred)
        else:
            if self.is_img_day:
                self.update_confusion_matrix_daytime(best_pred, gt_label, output_head, best_pred)
            else:
                self.update_confusion_matrix_nighttime(best_pred, gt_label, output_head, best_pred)

    def predict_vehicle(self, crop, row_index, country):
        if self.model_dims:
            crop = cv2.resize(src=crop, dsize=self.model_dims, interpolation=cv2.INTER_AREA)

        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = crop.astype(np.float32)
        crop = np.expand_dims(crop, axis=0)
        crop = np.transpose(crop, axes=[0, 3, 1, 2])

        # Inference Section
        ort_inputs = {self.session.get_inputs()[0].name: crop, self.session.get_inputs()[1].name: np.array([country])}  # Prepare the inputs for the inference
        start_time = time.time()
        ort_outputs = self.session.run(None, ort_inputs)  # Run inference and get outputs from inputs
        end_time = time.time()
        inference_time = end_time - start_time
        output_dict = {}

        # These 2 following lines should be removed when the issue of missing_plate and vehicle is solved
        # indexes_to_remove = [4, 6]
        # indexes_to_remove = [6]
        # ort_outputs = [ort_outputs[i] for i in range(len(ort_outputs)) if i not in indexes_to_remove]

        for i in range(len(ort_outputs)):
            output_head = self.classifier_output_heads[i]
            head_labels = self.cls_labels[output_head]
            curr_head_dict = {}
            for z, val in enumerate(ort_outputs[i][0]):
                # missing_plate & vehicle have only 2 values so there is no index value more than 2
                if output_head in ['missing_plate', 'vehicle'] and z == 2:
                    break
                curr_label = head_labels[str(z)]
                curr_head_dict[curr_label] = val
            output_dict[output_head] = {}
            output_dict[output_head]['full_array'] = curr_head_dict
            # Get the best prediction
            best_pred = max(curr_head_dict, key=curr_head_dict.get)
            output_dict[output_head]['best_pred'] = best_pred

            # Retrieve and scale the confidence score
            confidence_score = output_dict[output_head]["full_array"].get(best_pred, None)
            # if output_head in ['missing_plate', 'vehicle']:
            #     confidence_score = confidence_score * 10000 if confidence_score is not None else None

            # confidence_score = output_dict[output_head]["full_array"].get(best_pred, None)
            # if confidence_score is not None:
            #     # Scale the confidence score by 100 for better readability, if needed
            #     confidence_score = confidence_score * 10000 if output_head == 'missing_plate' else confidence_score

            # confidence_score = output_dict[output_head]["full_array"].get(best_pred, None)
            self.update_raw_dataset(output_head, best_pred, confidence_score, row_index, inference_time)
        return output_dict

    def create_benchmark_daytime(self):
        # *******Calculate metrics (precision, recall, f1-score) for each head
        metrics_dict = {}
        for head in self.classifier_output_heads:
            tp = self.tp_dict_daytime[head]
            fp = self.fp_dict_daytime[head]
            tn = self.tn_dict_daytime[head]
            fn = self.fn_dict_daytime[head]

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

            metrics_dict[head] = {
                'True Positives': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatives': fn,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1_score,
                'Accuracy': accuracy,
                'total_accuracy_based_on_all_heads': ''
            }

        # *******Calculate overall accuracy by using the total number of correct predictions and total predictions
        total_correct_predictions = 0
        total_predictions = 0
        for head in self.classifier_output_heads:
            total_correct_predictions += self.tp_dict_daytime[head]
            total_predictions += self.tp_dict_daytime[head] + self.fp_dict_daytime[head]

        overall_accuracy = total_correct_predictions / total_predictions if total_predictions != 0 else 0
        metrics_dict['bodytype']['total_accuracy_based_on_all_heads'] = overall_accuracy
        self.benchmark_results['accuracy_data_daytime'] = pd.DataFrame(metrics_dict)
        self.benchmark_results['raw_data'] = self.test_data

        # *******Confusion matrix for each head
        confusion_matrix_dfs = []
        for head in self.classifier_output_heads:
            predicted_labels = self.predicted_labels_dict_daytime[head]
            ground_truth_labels = self.ground_truth_labels_dict_daytime[head]
            confusion_matrix = pd.crosstab(pd.Series(ground_truth_labels, name='Actual'),
                                           pd.Series(predicted_labels, name='Predicted'))

            # Append the confusion matrix DataFrame to the list
            confusion_matrix_dfs.append(confusion_matrix)
            key_name = 'confusion_matrix_' + str(head) + '_daytime'
            self.benchmark_results[key_name] = confusion_matrix

        # *******Calculate detailed accuracy for each value of each head
        accuracy_dict = {}
        for head, categories in self.accuracy_tracker_daytime.items():
            head_accuracy = {}
            for category, correct in categories['correct_predictions'].items():
                total = categories['total_count'][category]
                accuracy = correct / total if total != 0 else 0
                head_accuracy[category] = {
                    'accuracy': accuracy,
                    'total_count': total,
                    'correct_predictions': correct
                }

            accuracy_dict[head] = head_accuracy

        detailed_accuracy_dict = accuracy_dict
        result_df = pd.DataFrame()
        for head, values in detailed_accuracy_dict.items():
            # Create a DataFrame from the head_accuracy dictionary
            accuracy_df = pd.DataFrame.from_dict(values, orient='index').reset_index()
            accuracy_df.rename(columns={'index': head,
                                        'accuracy': head + '_accuracy',
                                        'total_count': head + '_total_count',
                                        'correct_predictions': head + '_correct_predictions'}, inplace=True)

            result_df = pd.concat([result_df, accuracy_df], axis=1)
        self.benchmark_results['detailed_accuracy_daytime'] = result_df

    def create_benchmark_nighttime(self):
        # *******Calculate metrics (precision, recall, f1-score) for each head
        metrics_dict = {}
        for head in self.classifier_output_heads:
            tp = self.tp_dict_nighttime[head]
            fp = self.fp_dict_nighttime[head]
            tn = self.tn_dict_nighttime[head]
            fn = self.fn_dict_nighttime[head]

            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

            metrics_dict[head] = {
                'True Positives': tp,
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatives': fn,
                'Precision': precision,
                'Recall': recall,
                'Accuracy': accuracy,
                'F1-Score': f1_score,
                'total_accuracy_based_on_all_heads': ''
            }

        # *******Calculate overall accuracy by using the total number of correct predictions and total predictions
        total_correct_predictions = 0
        total_predictions = 0
        for head in self.classifier_output_heads:
            total_correct_predictions += self.tp_dict_nighttime[head]
            total_predictions += self.tp_dict_nighttime[head] + self.fp_dict_nighttime[head]

        overall_accuracy = total_correct_predictions / total_predictions if total_predictions != 0 else 0
        metrics_dict['bodytype']['total_accuracy_based_on_all_heads'] = overall_accuracy

        self.benchmark_results['accuracy_data_nighttime'] = pd.DataFrame(metrics_dict)
        self.benchmark_results['raw_data'] = self.test_data

        # *******Confusion matrix for each head
        confusion_matrix_dfs = []
        for head in self.classifier_output_heads:
            predicted_labels = self.predicted_labels_dict_nighttime[head]
            ground_truth_labels = self.ground_truth_labels_dict_nighttime[head]
            confusion_matrix = pd.crosstab(pd.Series(ground_truth_labels, name='Actual'),
                                           pd.Series(predicted_labels, name='Predicted'))

            # Append the confusion matrix DataFrame to the list
            confusion_matrix_dfs.append(confusion_matrix)
            key_name = 'confusion_matrix_' + str(head) + '_nighttime'
            self.benchmark_results[key_name] = confusion_matrix

        # *******Calculate detailed accuracy for each value of each head
        accuracy_dict = {}
        for head, categories in self.accuracy_tracker_nighttime.items():
            head_accuracy = {}
            for category, correct in categories['correct_predictions'].items():
                total = categories['total_count'][category]
                accuracy = correct / total if total != 0 else 0
                head_accuracy[category] = {
                    'accuracy': accuracy,
                    'total_count': total,
                    'correct_predictions': correct
                }

            accuracy_dict[head] = head_accuracy

        detailed_accuracy_dict = accuracy_dict
        result_df = pd.DataFrame()
        for head, values in detailed_accuracy_dict.items():
            # Create a DataFrame from the head_accuracy dictionary
            accuracy_df = pd.DataFrame.from_dict(values, orient='index').reset_index()
            accuracy_df.rename(columns={'index': head,
                                        'accuracy': head + '_accuracy',
                                        'total_count': head + '_total_count',
                                        'correct_predictions': head + '_correct_predictions'}, inplace=True)

            result_df = pd.concat([result_df, accuracy_df], axis=1)
        self.benchmark_results['detailed_accuracy_nighttime'] = result_df

    def store_results(self, output_loc):
        # Ensuring that benchmark_results dictionary is not empty
        if not self.benchmark_results:
            print("No data available in self.benchmark_results to save!")
            return None

        bad_data_df = pd.DataFrame(self.bad_data_logger.bad_data)
        self.benchmark_results['bad_data_log'] = bad_data_df

        image_embedding_df = pd.DataFrame(self.image_embedding_collector)
        self.benchmark_results['image_embeddings'] = image_embedding_df

        today = pd.to_datetime('today').strftime('%m-%d-%y')
        now = pd.to_datetime('today').strftime('%H_%M')

        # Creating timestamped folder name
        local_folder_name = today + '_' + now
        full_path = os.path.join(output_loc, local_folder_name)

        # Creating directory if it doesn't exist
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f"Created folder: {full_path}")
        else:
            print(f"Folder already exists: {full_path}")

        # Saving each dataframe in the benchmark_results to CSV
        for key, val in self.benchmark_results.items():
            if val.empty:
                print(f"No data to save in {key} csv file")
                continue

            file_name = f'{key}_{today}_{now}.csv'
            file_path = os.path.join(full_path, file_name)
            val.to_csv(file_path, sep=',', header=True, encoding='utf-8', quotechar='"')
            print(f'Storing Benchmark Data at: {file_path}')

        return full_path

# def make_parser():
#     parser = argparse.ArgumentParser("rekor Vehicle classifier benchmark")
#     parser.add_argument(
#         "--benchmark_tag",
#         type=str,
#         required=True,
#         help="Specify a string tag for the benchmark run",
#     )
#
#     parser.add_argument(
#         "--alpr_runtime",
#         type=str,
#         required=True,
#         help="Comma-delimited list of S3 folders for the ALPR runtime volume mount to run benchmarking e.g. 4.1.1,5.0.1",
#     )
#
#     parser.add_argument(
#         "--triton_service_url",
#         type=str,
#         default="",
#         help="Specify the triton_clipzeroshot service_url like http://3.89.217.59:8000",
#     )
#
#     parser.add_argument(
#         "--bench_csv_s3_path",
#         type=str,
#         default="",
#         help="Location of benchmark csv file in S3",
#     )
#
#     return parser
#

def main():
    # def server_input_settings():
    # is_local_test = False
    # message_dict = {}
    # args = make_parser().parse_args()
    #
    # debug = app_settings.config_data.debug
    # log_.info(f'Debug: {debug}')
    #
    # store_s3 = app_settings.config_data.store_s3
    # log_.info(f'Storing in S3: {store_s3}')
    #
    # tag = args.benchmark_tag
    # log_.info(f'Benchmark Tag: {tag}')
    # message_dict['BenchmarkTag'] = tag
    #
    # alpr_runtime_list = args.alpr_runtime.split(',')
    # log_.info(f'Model List: {alpr_runtime_list}')
    # message_dict['ModelVersionList'] = alpr_runtime_list
    #
    # bucket = app_settings.config_data.aws.output_bucket
    #
    # log_.info(f'Bucket: {bucket}')
    # message_dict['StorageBucket'] = bucket
    #
    # triton_service_url = args.triton_service_url
    #
    # csv_query = None
    # bench_csv_s3_path = args.bench_csv_s3_path
    # if bench_csv_s3_path != "":
    #     log_.info(f'Bench CSV Path: {bench_csv_s3_path}')
    #     message_dict['BenchS3Path'] = bench_csv_s3_path
    #     csv_query = '1'
    #
    # model_locations = retrieve_models()
    # log_.info(f'Model Locations: {model_locations}')
    # path_to_model = (model_locations['alpr'][args.alpr_runtime]['runtime_data'] +
    #                  '/vehicleclassifier/vehicle_x.onnx')
    # path_to_jsonfile = (model_locations['alpr'][args.alpr_runtime]['runtime_data'] +
    #                     '/vehicleclassifier/vehicle_labels.json')


    # metadata_file_name = 'benchmarking_dataset_100_samples.csv'
    metadata_file_name = '300_samples_out_of_full_test_set_changed_missingplates.csv'
    # metadata_file_name = '10samples_for_missing_plate_tes.csv'

    path_to_metadata = ('/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_metadata/' +
                        metadata_file_name)

    # paths to model
    # model_path = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_model_results/old_model/'
    # path_to_model = model_path + 'vehicle_classifier.onnx'
    # path_to_jsonfile = model_path + 'vehicle_labels.json'
    # #
    model_path = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_model_results/new_model_densent121/'
    path_to_model = model_path + 'classifier.onnx'
    path_to_jsonfile = model_path + 'vehicle_labels.json'

    # model_path = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_model_results/model_trained_with_1000samples_updated_missing_plate/'
    # path_to_model = model_path + 'classifier.onnx'
    # path_to_jsonfile = model_path + 'classifier.json'


    # model_path = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_model_results/new_model_local_missing_plate_changed/'
    # path_to_model = model_path + 'classifier.onnx'
    # path_to_jsonfile = model_path + 'classifier.json'

    cropped_path = '/home/elham/projects/alpr_vc_project/input_data/crops2/'
    image_loc = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/images2/'

    output_loc = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/01_output_results/'

    # triton_service_url = ''  # for when I don't want to call triton service
    triton_service_url = 'http://3.83.123.147:8000'  # the ip for when I launch from home
    # triton_service_url = 'http://10.100.13.55:8000'  # the private ip for your script in the cloud

    trition_day_night_counter = 0
    successful_images = []  # List to store filenames of successfully processed images
    try:
        am = AlprModels()
        am.get_metadata_csv(path_to_metadata)
        am.init_vehicle_classifier(model_path=path_to_model, json_path=path_to_jsonfile)
        empty_image_counter = 0
        empty_required_fields_counter = 0
        metadata_length = len(am.test_data)
        processed_vehicle_counter = 0
        for row_index, row in am.test_data.iterrows():
            try:
                processed_vehicle_counter += 1
                # read required fields and normalize them
                vehicle_id = row['vehicle_id']
                # if vehicle_id == 'vehicle-1572369340533-rlfcc':
                #     print('catched test sample')
                vehicle_bbox = ""
                if pd.isna(row['vehicle_bbox_machine']):
                    vehicle_bbox = row['vehicle_bbox']
                else:
                    vehicle_bbox = row['vehicle_bbox_machine']
                if pd.isna(row['image_file']) or pd.isna(vehicle_bbox):
                    empty_required_fields_counter += 1
                    print(f'vehicle_id "{vehicle_id}" has empty value in fields image_file or vehicle_bbox_machine')
                    continue

                def format_prefix(prefix):
                    """
                    Format the image_prefix to ensure single-digit prefixes are represented with leading zero.
                    """
                    if str(prefix).isdigit() and len(str(prefix)) == 1:
                        return f'0{prefix}'
                    return prefix

                missing_plate = row['missing_plate']
                # if missing_plate == 'no':
                #     print('yes')
                image_prefix = format_prefix(row['image_prefix'])
                image_file_path = image_loc + row["image_file"]
                image_file_name = row["image_file"]
                vehicle_bbox = vehicle_bbox.strip('[]')  # Remove square brackets
                bbox_int_list = [int(x) for x in vehicle_bbox.split(',')]  # Create a list of integers

                country = am.get_region_code(row['country_code'])
                image = am.read_image(image_file_path)

                # if triton_service_url != "":

                if vehicle_id in o3_alpr_triton_day_night_results.day_night_dict:
                    am.is_img_day = o3_alpr_triton_day_night_results.day_night_dict[vehicle_id]
                else:
                    # am.is_img_day = True  # {True = day , False = night}
                    am.is_img_day, img_embedding = alpr_daynight_classification.day_night_classification(
                        image_file_path,
                        image_file_name,
                        triton_service_url)
                    am.image_embedding_collector.append(
                        {"vehicle_id": vehicle_id, "img_file": image_file_path, "img_embedding": img_embedding,
                         'is_img_day': am.is_img_day})

                if image is None or image.size == 0:
                    # Skip processing if the image is empty
                    print(f'Skipping empty image: {image_file_path}')
                    empty_image_counter += 1
                    continue
                # Image cropping and prediction
                crop = am.crop_vehicle(image_file_name, vehicle_id, image, bbox_int_list)
                if len(crop) != 0:
                    am.predict_vehicle(crop, row_index, country)
                    successful_images.append(image_file_path)  # Store filename if processed successfully
                    print(f"Successfully predicted the vehicle_id: {vehicle_id} - image_path: {image_file_path} - {processed_vehicle_counter} out of {metadata_length} samples have been processed.")
                else:
                    print(f'Error processing image {image_file_path}')

            except Exception as e:
                print(f'Error processing image {image_file_path}: {str(e)}')

        # Evaluate how accurate the results are
        print(f'number of empty images:: {empty_image_counter}')
        print(f'number of empty images:: {empty_required_fields_counter}')
        am.create_benchmark_daytime()
        am.create_benchmark_nighttime()
        output_loc = am.store_results(output_loc)
        print('benchmarking finished successfully!')

    except Exception as e:
        message_str = '\n\nECS Task Failed \nPipeline Source: alpr \nPipeline Stage: vehicle-classifier-benchmark \n\n'
        print(message_str)
        print(e)


main()







