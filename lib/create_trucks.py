from cloud_data_manager import s3_manager
import pandas as pd
from io import StringIO,BytesIO
from tqdm import tqdm
import numpy as np


#
# frame_metadata_cols = [
#     'frame_num',
#     'frame_num_original',
#     'processed_time',
#     'frame_s3_key',
#     'class_0_count',
#     'class_0_mean_confidence',
#     'class_0_mean_boxarea',
#     'class_0_max_objectid',
#     'class_1_count',
#     'class_1_mean_confidence',
#     'class_1_mean_boxarea',
#     'class_1_max_objectid',
#     'class_2_count',
#     'class_2_mean_confidence',
#     'class_2_mean_boxarea',
#     'class_2_max_objectid',
#     'class_3_count',
#     'class_3_mean_confidence',
#     'class_3_mean_boxarea',
#     'class_3_max_objectid',
#     'class_4_count',
#     'class_4_mean_confidence',
#     'class_4_mean_boxarea',
#     'class_4_max_objectid',
#     'class_5_count',
#     'class_5_mean_confidence',
#     'class_5_mean_boxarea',
#     'class_5_max_objectid',
#     'class_6_count',
#     'class_6_mean_confidence',
#     'class_6_mean_boxarea',
#     'class_6_max_objectid',
#     'class_7_count',
#     'class_7_mean_confidence',
#     'class_7_mean_boxarea',
#     'class_7_max_objectid',
#     'class_8_count',
#     'class_8_mean_confidence',
#     'class_8_mean_boxarea',
#     'class_8_max_objectid',
#     'class_9_count',
#     'class_9_mean_confidence',
#     'class_9_mean_boxarea',
#     'class_9_max_objectid',
#     'class_10_count',
#     'class_10_mean_confidence',
#     'class_10_mean_boxarea',
#     'class_10_max_objectid',
#     'class_11_count',
#     'class_11_mean_confidence',
#     'class_11_mean_boxarea',
#     'class_11_max_objectid',
#     'class_12_count',
#     'class_12_mean_confidence',
#     'class_12_mean_boxarea',
#     'class_12_max_objectid',
#     'inference_time_ms',
#     'allclass_mean_confidence',
#     'allclass_mean_boxarea_pixels',
#     'width_original',
#     'height_original',
#     'fps_original',
#     'fps_downsample',
#     'source_video',
#     'source_s3_bucket',
#     'source_s3_key',
#     'destination_s3_location',
#     'destination_s3_data_location',
#     'streaming_flag',
#     'batch_name',
#     'object_tracking_model_onnx',
#     'sample_clip_not_fps_normalized_s3_key',
#     'hue',
#     'value',
#     'saturation',
#     'focus_measure',
#     'mean_red',
#     'mean_green',
#     'mean_blue'
# ]
#
# s3m = s3_manager()
# bucket = 'rekor-tracker-data'
#
# folder_list = [
# 'objecttracking-external-rekor-ccs-fullbenchmark-video1-final-20230424-2206',
# # 'objecttracking-external-rekordata-batch-1-20220926-0842',
# # 'objecttracking-external-rekordata-batch1-i10-11am-20220929-2318',
# # 'objecttracking-external-rekordata-batch2-i10-830pm-20221006-0037',
# # 'objecttracking-external-rekordata-batch4-i10-8am-20221011-0122',
# # 'objecttracking-external-rekordata-batch5-3sitesmixed-take2-20221114-2246',
#
# ]
#
# count=0
# for folder in folder_list:
#     print(folder)
#     key = 'video/frame_metadata/' + folder + '/'
#     file_list = s3m.list_bucket_objects(bucket,key,'.csv')
#     file_loop = tqdm(file_list, total=len(file_list), leave=False)
#
#     for file in file_loop:
#
#         df = s3m.get_bucket_object_df(bucket=bucket,
#                                           key=file)
#
#         for col in frame_metadata_cols:
#             if col not in df.columns:
#                 df[col] = np.NaN
#
#         df2 = df[frame_metadata_cols]
#
#
#         csv_buf = StringIO()
#         df2.to_csv(csv_buf, sep=',', header=True, index=False,encoding='utf-8',quotechar='"')
#         csv_buf.seek(0)
#         s3m.s3client.put_object(Bucket=bucket,
#                                  Key=file,
#                                  Body=csv_buf.getvalue())
#





folder_list = [
    'objecttracking-sts-external-batch-1-20220627-1518',
    'objecttracking-sts-external-batch-2-20220707-1429',
    'objecttracking-sts-external-batch-4-resubmission',
    'objecttracking-sts-external-batch-5-trucks-top30-20220816-1935',
    'objecttracking-sts-external-batch-8-nighttimefix2-60-20220824-0136',
    'objecttracking-sts-external-batch-9-totalvolume2-60clips-20220831-1744',
    'objecttracking-sts-external-batch-resubmit-1-26clips-20220831-1529'
]



s3m = s3_manager()
bucket = 'sts-classifier-data'

count=0
for folder in folder_list:
    print(folder)
    key = 'video-annotations/training_data/versioned_datasets/tracking/' + folder + '/'
    file_list = s3m.list_bucket_objects(bucket,key,'.csv')
    file_loop = tqdm(file_list, total=len(file_list), leave=False)
    for file in file_loop:
        df = s3m.get_bucket_object_df(bucket=bucket,
                                              key=file)

        filename = file.split('/')[-1]
        new_file = 'video-annotations/training_data/tracking_annotations/' + folder + '/' + filename

        csv_buf = StringIO()
        df.to_csv(csv_buf, sep='|', header=True, index=False,encoding='utf-8',quotechar='"')
        csv_buf.seek(0)
        s3m.s3client.put_object(Bucket=bucket,
                                Key=new_file,
                                Body=csv_buf.getvalue())

