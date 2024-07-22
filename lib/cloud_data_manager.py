import boto3
import os
from tqdm import tqdm
import pdb
import pandas as pd
import time
import io

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#REQUIRES AWS CLI LOGIN OR CREDENTIALS PROVIDED TO BOTO VIA SCRIPT/IAM ROLE
#ROLE MUST HAVE AWS PERMISSIONS TO:
# 1)  READ/WRITE FOR ANY S3 BUCKETS (inputs/outputs if not provided by workgroup) AND
# 2)  ATHENA QUERY EXECUTION/MONITORING
# 3)  GLUE DATA CATALOG???


# REMEMBER THAT ATHENA IS CHARGED BY TOTAL DATA SCANNED SO BE MINDFUL OF QUERY SIZE
# AND UTILIZE LIMITS IN THE SQL

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class RekognitionLabel:
    """Encapsulates an Amazon Rekognition label."""
    def __init__(self, label, timestamp=None):
        """
        Initializes the label object.

        :param label: Label data, in the format returned by Amazon Rekognition
                      functions.
        :param timestamp: The time when the label was detected, if the label
                          was detected in a video.
        """
        self.name = label.get('Name')
        self.confidence = label.get('Confidence')
        self.instances = label.get('Instances')
        self.parents = label.get('Parents')
        self.timestamp = timestamp

    def to_dict(self):
        """
        Renders some of the label data to a dict.

        :return: A dict that contains the label data.
        """
        rendering = {}
        if self.name is not None:
            rendering['name'] = self.name
        if self.timestamp is not None:
            rendering['timestamp'] = self.timestamp
        return rendering

class rekognition_manager():

    def __init__(self,aws_access_key_id='',
                      aws_secret_key='',
                      aws_session_token=''):

        if (aws_access_key_id!=''):
            self.aws_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                            aws_secret_access_key=aws_secret_key,
                                            aws_session_token=aws_session_token)
        else:
            self.aws_session = boto3.Session()

        self.rekogclient = self.aws_session.client('rekognition')
        pass


    def detect_labels(self, bucket,
                            image_key,
                      max_labels=30):
        """
        Detects labels in the image. Labels are objects and people.

        :param max_labels: The maximum number of labels to return.
        :return: The list of labels detected in the image.
        """
        image_name = image_key.split('/')[-1]


        labels = []

        # try:
        response = self.rekogclient.detect_labels(
                Image={'S3Object':{
                                    'Bucket':bucket,
                                    'Name':image_key
                                  }
                      },
                MaxLabels=max_labels
        )


        print(response)

        label_dict = {}
        for resp in response['Labels']:
            label_dict[resp['Name']] = {}


            if len(resp['Instances']):
                inst_count=0
                for inst in resp['Instances']:
                    label_dict[resp['Name']][inst_count] = {}
                    label_dict[resp['Name']][inst_count]['bb'] = inst['BoundingBox']
                    label_dict[resp['Name']][inst_count]['confidence'] = inst['Confidence']
                    label_dict[resp['Name']][inst_count]['parents'] = inst['Parents']
            else:
                label_dict[resp['Name']]['confidence'] = resp['Confidence']

        # labels = [RekognitionLabel(label) for label in response['Labels']]
        # pdb.set_trace()
        # print("Found %s labels in {}".format(len(labels), image_name))
        # # except:
        # #     print("Couldn't detect labels in {}".format(image_name))
        #
        # for label in labels[:3]:
        #     print(label.to_dict())
        return label_dict


class athena_manager():
    def __init__(self,aws_access_key_id='',
                      aws_secret_key='',
                      aws_session_token=''):
        '''

        :param aws_access_key_id:
        :param aws_secret_key:
        :param aws_session_token:
        '''
        if (aws_access_key_id!=''):
            self.aws_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                            aws_secret_access_key=aws_secret_key,
                                            aws_session_token=aws_session_token)
        else:
            self.aws_session = boto3.Session()

        self.athenaclient = self.aws_session.client('athena')
        pass

    def run_query(self,sql_query,
                       glue_database,
                       athena_workgroup='',
                       output_bucket_location='',
                       max_attempts=5

                  ):
        '''

        :param sql_query: Athena/Presto SQL query to run - REQUIRED
        :param glue_database: Glue database containing required tables - REQUIRED
        :param athena_workgroup: Athena WorkGroup to run query
        :param output_bucket_location: S3 bucket location of outputs (REQUIRED if not using WorkGroup)
        :return: S3 location of output csv file
        '''
        if (output_bucket_location):
            query_id = self.athenaclient.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={
                    'Database': glue_database
                },
                ResultConfiguration={'OutputLocation': output_bucket_location},

            )
        elif (athena_workgroup):
            query_id = self.athenaclient.start_query_execution(
                QueryString=sql_query,
                QueryExecutionContext={
                    'Database': glue_database
                },
                WorkGroup=athena_workgroup
            )
        else:
            return
        time.sleep(10)


        total_attempts = 0
        while True:
            query_execution_status = self.athenaclient.get_query_execution(QueryExecutionId=query_id['QueryExecutionId'])
            status = query_execution_status['QueryExecution']['Status']['State']
            if status=='SUCCEEDED':
                result_output = query_execution_status['QueryExecution']['ResultConfiguration']['OutputLocation']
                print(result_output)
                break
            else:
                if total_attempts>max_attempts:
                    result_output = ''
                    self.athenaclient.stop_query_execution(QueryExecutionId=query_id['QueryExecutionId'])
                    print('Stopping Athena Query - error in execution')
                    break
                else:

                    total_attempts+=1
                    print('Attempt {} / {} - Waiting additional 10s for Athena Query to complete'.format(total_attempts,max_attempts))
                    time.sleep(10)

        return result_output




class s3_manager():
    def __init__(self,aws_access_key_id='',
                      aws_secret_key='',
                      aws_session_token=''):
        '''

        :param aws_access_key_id:
        :param aws_secret_key:
        :param aws_session_token:
        '''
        if (aws_access_key_id!=''):
            self.aws_session = boto3.Session(aws_access_key_id=aws_access_key_id,
                                            aws_secret_access_key=aws_secret_key,
                                            aws_session_token=aws_session_token)
        else:
            self.aws_session = boto3.Session()

        self.s3client = self.aws_session.client('s3')
        self.total_files_moved = 0
        self.file_download_limit = 100000
        pass

    def get_bucket_object_df(self,bucket,key,delimiter=None):
        '''
        Retrieve pandas dataframe of csv file stored in S3

        :param bucket: s3 bucket of csv file
        :param key: s3 key of csv file
        :return: pandas dataframe of csv file
        '''
        obj = self.s3client.get_object(Bucket=bucket, Key=key)

        if delimiter != None:
            csv_df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8',delimiter=delimiter)
        else:
            csv_df = pd.read_csv(io.BytesIO(obj['Body'].read()), encoding='utf-8')
        return csv_df

    def get_bucket_object(self,bucket,key):
        '''
        Retrieve bytes of object stored in S3
        :param bucket: s3 bucket of object
        :param key: s3 of object
        :return: bytes string of object's contents
        '''
        obj = self.s3client.get_object(Bucket=bucket, Key=key)
        return io.BytesIO(obj['Body'].read())

    def download_bucket_objects(self,bucket,
                                 prefix,
                                local_filepath,
                                text_filter=''):
        '''

        download all files from a bucket given a folder prefix

        :param bucket: s3 bucket of files to download
        :param prefix: s3 prefix (subfolder/location) of files to download
        :param local_filepath: local location to download to
        :param text_filter: string filter on filename
        :return:
        '''
        paginator = self.s3client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        out_dir = os.path.join(local_filepath, prefix)
        os.makedirs(out_dir, exist_ok=True)
        page_count=1
        for page in pages:
            file_listd = [x['Key'] for x in page['Contents']]
            file_loop = tqdm(file_listd, total=len(file_listd), leave=False)
            for filed in file_loop:
                if text_filter:
                    if text_filter not in filed.split('/')[-1]:
                        continue
                full_path = os.path.join(out_dir, filed.split('/')[-1])
                try:
                    if self.total_files_moved<=self.file_download_limit:
                        self.s3client.download_file(bucket,filed,full_path)
                        self.total_files_moved+=1
                    else:
                        print('file download limit reached -> breaking...')
                        pass
                except Exception as e:
                    print('Error - {} - continuing...\n'.format(e))
                    continue
            page_count+=1
        print('total files Downloaded: {} '.format(self.total_files_moved))
        pass

    def list_bucket_objects(self,bucket,
                             prefix='',
                             text_filter=''):
        '''
        Lists all objects in a S3 bucket given a S3 prefix

        :param bucket: s3 bucket of files to list
        :param prefix: s3 prefix (subfolder/location) of files to list
        :param text_filter: string filter on filename
        :return:
        '''
        paginator = self.s3client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        total_list = []
        page_count = 1
        for page in pages:
            if 'Contents' in page.keys():
                if text_filter:
                    file_listd = [x['Key'] for x in page['Contents'] if text_filter in x['Key']]
                else:
                    file_listd = [x['Key'] for x in page['Contents']]
            else:
                print('Error in bucket {} prefix {} page {}'.format(bucket,prefix,page))
                page_count+=1
                continue
            total_list+=file_listd

            page_count+=1

        return total_list