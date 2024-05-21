import numpy as np
from sagemaker import get_execution_role
import boto3
import pandas as pd
import json


bucket_name = 'chianglab-dataderivatives'
file_path = "mimic-iv-ed-2.2/text_repr.json"

s3 = boto3.resource('s3')
content_object = s3.Object(bucket_name, file_path)
file_content = content_object.get()['Body'].read().decode('utf-8')
json_content = json.loads(file_content)
df = pd.DataFrame(json_content).T
df['medrecon'] = df['medrecon'].fillna("The patient did not receive any medications recorded.")
df['pyxis'] = df['pyxis'].fillna("The patient did not receive any medications.")
df['vitals'] = df['vitals'].fillna("The patient had no vitals recorded.")
df['codes'] = df['codes'].fillna("The patient has no diagnostic codes recorded.")
df = df.drop("admission", axis=1)
df = df.drop("discharge", axis=1)
df = df.drop("eddischarge_category", axis=1)
df = df.drop("eddischarge", axis=1)
df['subject_id'] = df['arrival'].astype(str).str.split().str[1].replace(",", " ", regex=True).to_list()

# Additional data type setting
df['stay_id'] = df.index
df['stay_id'] = df['stay_id'].astype('int64')
df['subject_id'] = df['subject_id'].astype('int64')

# Adding hadm_id to dataset
edstays = pd.read_csv("./edstays.csv.gz")
edstays = edstays.fillna(0)
edstays['hadm_id'] = edstays['hadm_id'].astype(int)
edstays = edstays[['stay_id', 'hadm_id']]
df2 = pd.merge(df, edstays, on='stay_id')

# Loading antibiotics data
antibiotics = pd.read_csv("./antibiotic_labels.csv")
antibiotics['hadm_id'] = antibiotics['hadm_id'].astype(int)
df_final = pd.merge(df2, antibiotics, on='hadm_id', how='inner')
df_final = df_final.fillna(0)
df = df_final
df.to_csv("antibiotics.csv")