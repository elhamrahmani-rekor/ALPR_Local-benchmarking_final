import pandas as pd
import pdb

datad = pd.read_csv('/home/wmills/Downloads/dotclassifier_train_sts_stscrops_flatdist_800k_10percsample_allsites_05-10-2023_01-13.csv')

for name,group in datad.groupby('imagefilename'):
    if len(group)>1:
        pdb.set_trace()
