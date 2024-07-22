import os
import base64
import json
import requests
import numpy as np
import cv2

image_dir = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/input_data/day_night_images/'
day_time_image_dir_mix = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/results/zeroshot_day_night_result/day_mix'
night_time_image_dir_mix = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/results/zeroshot_day_night_result/night_mix'
errors_dir = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/results/zeroshot_day_night_result/test_mix'
daytime_hsv_dir = '/home/elham/projects/alpr_vc_project/alpr_benchmarking/results/hsv_day_night_result/day'
nighttime_hsv_dir ='/home/elham/projects/alpr_vc_project/alpr_benchmarking/results/hsv_day_night_result/night'


# Function to clean base64 encoded data
def clean_base64(data):  # Clean the base64 data using your clean_base64 function
    if ';base64,' in data:
        data = data.split(';base64,')[1]
    new_data = data.replace(' ', '').replace('\n', '')
    return new_data


def call_triton_server(service_url, selected_model, b64_image=None, text_input=None, model_option=None,):

    # Get Model information about input/output format
    response = requests.get(f"{service_url}/v2/models/{selected_model}")
    metadata = json.loads(response.text)

    clean_img = clean_base64(b64_image)
    batched_images = [clean_img]

    # To print all inputs/outputs
    inputs = []
    for input in metadata['inputs']:
        inputs.append(input)

    outputs = []

    for output in metadata['outputs']:
        outputs.append(output)

    texts = None
    if text_input:
        texts = text_input.split('|')
        if len(texts) != len(batched_images):
            raise Exception(f'Incorrect batch sizing - please check all inputs to verify array shapes and delimiters')

    model_options = None
    if model_option:
        model_options = model_option.split('|')
        if len(model_options) != len(batched_images):
            raise Exception(f'Incorrect batch sizing - please check all inputs to verify array shapes and delimiters')

    payload = {
        'inputs': [
            {
                'name': inputs[0]['name'],
                'shape': [len(batched_images), 1],
                'datatype': inputs[0]['datatype'],
                'data': batched_images
            }
        ]
    }

    if texts:
        text_payload = {
            'name': "text_input",
            'shape': [len(texts), 1],
            'datatype': inputs[1]['datatype'],
            'data': texts
        }
    else:
        text_payload = {
            'name': "text_input",
            'shape': [len(batched_images), 1],
            'datatype': inputs[1]['datatype'],
            'data': [" " for x in range(len(batched_images))]
        }

    payload['inputs'].append(text_payload)

    if model_options:
        options_payload = {
            'name': "model_option",
            'shape': [len(model_options), 1],
            'datatype': inputs[2]['datatype'],
            'data': model_options
        }
    else:
        options_payload = {
            'name': "model_option",
            'shape': [len(batched_images), 1],
            'datatype': inputs[2]['datatype'],
            'data': [" " for x in range(len(batched_images))]
        }

    payload['inputs'].append(options_payload)

    payload['outputs'] = outputs
    # Send POST request to Triton server
    response = requests.post(f'{service_url}/v2/models/{selected_model}/infer',
                             headers={'Content-Type': 'application/json'},
                             data=json.dumps(payload))

    # Instead of printing, we will return the reshaped_data
    if response.status_code == 200:
        result = json.loads(response.text)
        output_data = {}
        for outp in result['outputs']:
            reshaped_data = np.array(outp['data']).reshape(outp['shape'])
            output_data[outp["name"]] = reshaped_data.tolist()
        return output_data
    else:
        raise Exception(f'Failed to get a response. Status code: {response.status_code}, Response: {response.text}')

    return output_data


def call_triton_server_multi_model_support(service_url, selected_models, b64_image=None, text_inputs=None, model_options=None):
    results = []
    clean_img = clean_base64(b64_image) if b64_image else None

    if model_options is None:
        model_options = [" "] * len(selected_models)

    for model, text_input, model_option in zip(selected_models, text_inputs, model_options):
        # Get Model information about input/output format
        response = requests.get(f"{service_url}/v2/models/{model}")
        if response.status_code != 200:
            results.append(
                f"Failed to get model metadata. Status code: {response.status_code}, Response: {response.text}")
            continue

        metadata = json.loads(response.text)

        inputs = metadata['inputs']
        outputs = [out for out in metadata['outputs']]

        payload = {
            'inputs': [
                {
                    'name': inputs[0]['name'],
                    'shape': [1, 1],
                    'datatype': inputs[0]['datatype'],
                    'data': [clean_img]
                },
                {
                    'name': "text_input",
                    'shape': [1, 1],
                    'datatype': inputs[1]['datatype'],
                    'data': [text_input if text_input else " "]
                },
                {
                    'name': "model_option",
                    'shape': [1, 1],
                    'datatype': inputs[2]['datatype'],
                    'data': [model_option if model_option else " "]
                }
            ],
            'outputs': outputs
        }

        # Send POST request to Triton server
        response = requests.post(f'{service_url}/v2/models/{model}/infer',
                                 headers={'Content-Type': 'application/json'},
                                 data=json.dumps(payload))

        if response.status_code == 200:
            result = json.loads(response.text)
            output_data = {}
            for outp in result['outputs']:
                reshaped_data = np.array(outp['data']).reshape(outp['shape'])
                output_data[outp["name"]] = reshaped_data.tolist()
            results.append(output_data)
        else:
            results.append(f'Failed to get a response. Status code: {response.status_code}, Response: {response.text}')

    return results


def apply_hsv_evaluation(img, image_filename):
    hue_threshold = 4.50
    saturation_threshold = 15
    value_threshold = 103
    hue_plate = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hue_plate)
    v_sumed = np.sum(v)
    value = v_sumed / (float(img.shape[0] * img.shape[1]))
    s_sumed = np.sum(s)
    saturation = s_sumed / (float(img.shape[0] * img.shape[1]))
    h_sumed = np.sum(h)
    hue = h_sumed / (float(img.shape[0] * img.shape[1]))
    if hue > hue_threshold and saturation > saturation_threshold and value > value_threshold:
        is_img_day = True  # True means the image is day
    else:
        is_img_day = False  # False means the image is night

    return is_img_day


def generate_range(median):
    temp_list = []
    step = 0.01
    for i in range(-8, 9):
        number = median + i * step
        number = int(number * 100) / 100.0  # Truncate to two decimal places
        temp_list.append(number)
    return temp_list


def print_text_on_image(img_copy, save_path, probabilities):
    # probabilities[0] contains day confidence score and probabilities[1] contains night confidence score
    day_confidence_score_text = f"Probability[0]: {probabilities[0]:.2f}"
    night_confidence_score_text = f"Probability[1]: {probabilities[1]:.2f}"
    color1 = (255, 0, 0)  # Blue for probabilities[0]
    color2 = (0, 0, 255)  # Red for probabilities[1]

    # Put text on the image
    cv2.putText(img_copy, day_confidence_score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color1, 2)
    cv2.putText(img_copy, night_confidence_score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color2, 2)

    cv2.imwrite(save_path, img_copy)


def remove_images(folder_path):
    files = os.listdir(folder_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    for file in files:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Removed: {file_path}")
    print("All images removed.")


def clean_image_folders():
    remove_images(day_time_image_dir_mix)
    remove_images(night_time_image_dir_mix)
    remove_images(errors_dir)
    remove_images(daytime_hsv_dir)
    remove_images(nighttime_hsv_dir)


def day_night_classification(image_file_path, img_filename, triton_service_url):
    with open(image_file_path, 'rb') as image_file_path:
        image_data = image_file_path.read()  # Read the image file as binary data

        # required parameters for calling triton service
        base64_data = base64.b64encode(image_data).decode('utf-8')  # Convert binary data to base64
        models = ["clipzeroshot_ensemble_encoded", "clipembed_ensemble_encoded"]
        text_inputs = ["day,night", ""]
        model_options = [None, None]
        # image_analysis_result = call_triton_server_multi_model_support(service_url=triton_service_url,
        #                                            selected_models=models,
        #                                            b64_image=base64_data,
        #                                            text_inputs=text_inputs,
        #                                            model_options=model_options)

        models = 'clipzeroshot_ensemble_encoded'
        text_inputs = 'day,night'
        model_options = None
        image_analysis_result = call_triton_server(service_url=triton_service_url,
                                                                       selected_models=models,
                                                                       b64_image=base64_data,
                                                                       text_inputs=text_inputs,
                                                                       model_options=model_options)

        image_embeds = None
        probabilities = None

        for result in image_analysis_result:
            if 'probabilities' in result:
                probabilities = result['probabilities'][0]
            if 'image_embeds' in result:
                image_embeds = result['image_embeds']

        image_file = image_file_path.name
        img = cv2.imread(image_file)
        # probabilities = image_analysis_result['probabilities'][0]

        # Do zeroshot and hsv evaluation:
        if probabilities[0] > probabilities[1]:  # evaluation for daytime probabilities
            if probabilities[0] - probabilities[1] <= 0.20:
                is_img_day = apply_hsv_evaluation(img, img_filename)
            else:
                median_day = probabilities[0] / 2
                truncated_day_median = int(median_day * 100) / 100.0
                truncated_night_probabilities = int(probabilities[1] * 100) / 100.0
                day_median_range = generate_range(truncated_day_median)

                if truncated_night_probabilities in day_median_range:
                    is_img_day = apply_hsv_evaluation(img, img_filename)
                else:
                    is_img_day = True

        elif probabilities[1] > probabilities[0]:  # evaluation for nighttime probabilities
            if probabilities[1] - probabilities[0] <= 0.20:
                is_img_day = apply_hsv_evaluation(img, img_filename)
            else:
                median_night = probabilities[1] / 2
                truncated_night_median = int(median_night * 100) / 100.0
                truncated_day_probabilities = int(probabilities[0] * 100) / 100.0
                night_median_range = generate_range(truncated_night_median)

                if truncated_day_probabilities in night_median_range:
                    is_img_day = apply_hsv_evaluation(img, img_filename)
                else:
                    is_img_day = False
        elif probabilities[0] == probabilities[1]:
            is_img_day = apply_hsv_evaluation(img, img_filename)
    return is_img_day, image_embeds


image_counter = 0
triton_and_hsv_day_list = []
triton_and_hsv_night_list = []
is_img_day = True

print(f'Total number of images: {image_counter}')
print('triton_and_hsv_night_list:', triton_and_hsv_night_list)
print('triton_and_hsv_day_list:', triton_and_hsv_day_list)


