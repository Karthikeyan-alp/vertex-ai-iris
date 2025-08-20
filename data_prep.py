import os
from sklearn.datasets import load_iris
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DATA_PATH = os.getenv('DATA_PATH', 'data/iris.csv')

def upload_to_gcs(local_path, bucket_name, dest_blob):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    print(f'Uploaded {local_path} to gs://{bucket_name}/{dest_blob}')

if __name__ == '__main__':
    iris = load_iris(as_frame=True)
    df = iris.frame
    # keep target column name as 'target'
    local_csv = 'iris.csv'
    df.to_csv(local_csv, index=False)
    print(f'Saved local CSV: {local_csv} rows={len(df)}')
    upload_to_gcs(local_csv, BUCKET_NAME, DATA_PATH)
