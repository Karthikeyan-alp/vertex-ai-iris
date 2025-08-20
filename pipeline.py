# import os
# from dotenv import load_dotenv
# load_dotenv()

# PROJECT_ID = os.getenv('PROJECT_ID')
# REGION = os.getenv('REGION', 'us-central1')
# BUCKET_NAME = os.getenv('BUCKET_NAME')
# PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', f'gs://{BUCKET_NAME}/pipeline_root')
# GCS_CSV = os.getenv('DATA_PATH', 'data/iris.csv')
# GCS_CSV_URI = f'gs://{BUCKET_NAME}/{GCS_CSV}'

# from kfp import dsl, compiler
# from kfp.dsl import component

# BASE_IMAGE = 'python:3.10'
# PKGS = ['google-cloud-aiplatform==1.56.0']

# # -------------------- Dataset Component --------------------
# @component(base_image=BASE_IMAGE, packages_to_install=PKGS)
# def ensure_tabular_dataset(project: str, location: str, display_name: str, gcs_uri: str) -> str:
#     from google.cloud import aiplatform
#     aiplatform.init(project=project, location=location)
#     for d in aiplatform.TabularDataset.list():
#         if d.display_name == display_name:
#             print('Reusing dataset', d.resource_name)
#             return d.resource_name
#     ds = aiplatform.TabularDataset.create(display_name=display_name, gcs_source=[gcs_uri])
#     print('Created dataset', ds.resource_name)
#     return ds.resource_name

# # -------------------- Training Component --------------------
# @component(base_image=BASE_IMAGE, packages_to_install=PKGS)
# def train_automl(project: str, location: str, dataset_resource_name: str, target_col: str,
#                  model_display_name: str, budget_mnh: int) -> str:
#     from google.cloud import aiplatform
#     aiplatform.init(project=project, location=location)
#     ds = aiplatform.TabularDataset(dataset_resource_name)

#     job = aiplatform.AutoMLTabularTrainingJob(
#         display_name=model_display_name + '-job',
#         optimization_prediction_type='classification',
#         optimization_objective='minimize-log-loss'
#     )

#     model = job.run(
#         dataset=ds,
#         target_column=target_col,
#         model_display_name=model_display_name,
#         budget_milli_node_hours=budget_mnh,
#         sync=True
#     )

#     try:
#         metrics = model.evaluate()
#         print('Evaluation metrics:', metrics)
#     except Exception as e:
#         print('Eval skipped:', e)

#     return model.resource_name

# # -------------------- Deployment Component --------------------
# @component(base_image=BASE_IMAGE, packages_to_install=PKGS)
# def deploy_model(project: str, location: str, model_resource_name: str, endpoint_display_name: str,
#                  machine_type: str = 'n1-standard-2') -> str:
#     from google.cloud import aiplatform
#     aiplatform.init(project=project, location=location)
#     model = aiplatform.Model(model_resource_name)
#     endpoint = model.deploy(
#         deployed_model_display_name=endpoint_display_name,
#         machine_type=machine_type,
#         min_replica_count=1,
#         max_replica_count=1
#     )
#     print('Deployed endpoint', endpoint.resource_name)
#     return endpoint.resource_name

# # -------------------- Pipeline --------------------
# @dsl.pipeline(name='iris-automl-pipeline')
# def iris_pipeline(project: str = PROJECT_ID, location: str = REGION,
#                   dataset_display_name: str = 'iris-dataset', gcs_csv_uri: str = GCS_CSV_URI,
#                   target_column: str = 'target', model_display_name: str = 'iris-automl-model',
#                   endpoint_display_name: str = 'iris-endpoint', budget_mnh: int = 1000):

#     ds = ensure_tabular_dataset(project=project, location=location,
#                                 display_name=dataset_display_name, gcs_uri=gcs_csv_uri)
#     model = train_automl(project=project, location=location,
#                          dataset_resource_name=ds.output, target_col=target_column,
#                          model_display_name=model_display_name, budget_mnh=budget_mnh)
#     _ = deploy_model(project=project, location=location,
#                      model_resource_name=model.output, endpoint_display_name=endpoint_display_name)

# # -------------------- Main --------------------
# if __name__ == '__main__':
#     compiler.Compiler().compile(pipeline_func=iris_pipeline, package_path='iris_pipeline.json')
#     print('Compiled pipeline to iris_pipeline.json')

#     from google.cloud import aiplatform
#     aiplatform.init(project=PROJECT_ID, location=REGION)
#     job = aiplatform.PipelineJob(display_name='iris-automl-pipeline-run',
#                                  template_path='iris_pipeline.json',
#                                  pipeline_root=PIPELINE_ROOT,
#                                  parameter_values={'project': PROJECT_ID, 'location': REGION})
#     job.run(sync=False)
#     print('Submitted pipeline job (async).')



import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'us-central1')
BUCKET_NAME = os.getenv('BUCKET_NAME')
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', f'gs://{BUCKET_NAME}/pipeline_root')
GCS_CSV = os.getenv('DATA_PATH', 'data/iris.csv')
GCS_CSV_URI = f'gs://{BUCKET_NAME}/{GCS_CSV}'

from kfp import dsl, compiler
from kfp.dsl import component

BASE_IMAGE = 'python:3.10'
PKGS = ['scikit-learn', 'pandas', 'joblib', 'google-cloud-aiplatform==1.56.0']

# -------------------- Dataset Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def ensure_tabular_dataset(project: str, location: str, display_name: str, gcs_uri: str) -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    for d in aiplatform.TabularDataset.list():
        if d.display_name == display_name:
            print('Reusing dataset', d.resource_name)
            return d.resource_name
    ds = aiplatform.TabularDataset.create(display_name=display_name, gcs_source=[gcs_uri])
    print('Created dataset', ds.resource_name)
    return ds.resource_name

# -------------------- Custom Training Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def train_custom_model(gcs_csv_uri: str, target_col: str, model_gcs_path: str) -> str:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    from google.cloud import storage

    # Load dataset
    df = pd.read_csv(gcs_csv_uri)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Test accuracy: {acc}")

    # Save model locally
    local_model_file = '/tmp/model.joblib'
    joblib.dump(clf, local_model_file)

    # Upload to GCS
    client = storage.Client()
    bucket_name, *path_parts = model_gcs_path.replace('gs://','').split('/')
    path = '/'.join(path_parts)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_filename(local_model_file)
    print(f'Model uploaded to {model_gcs_path}')

    return model_gcs_path

# -------------------- Deployment Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def deploy_model(model_gcs_path: str, endpoint_display_name: str) -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=endpoint_display_name + '-model',
        artifact_uri=model_gcs_path,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-1:latest'
    )

    endpoint = model.deploy(
        deployed_model_display_name=endpoint_display_name,
        machine_type='n1-standard-2'
    )

    print(f'Deployed endpoint: {endpoint.resource_name}')
    return endpoint.resource_name

# -------------------- Pipeline --------------------
@dsl.pipeline(name='iris-custom-pipeline')
def iris_pipeline(dataset_display_name: str = 'iris-dataset',
                  gcs_csv_uri: str = GCS_CSV_URI,
                  target_column: str = 'target',
                  model_gcs_path: str = f'gs://{BUCKET_NAME}/models/iris_model.joblib',
                  endpoint_display_name: str = 'iris-endpoint'):

    ds = ensure_tabular_dataset(display_name=dataset_display_name, gcs_uri=gcs_csv_uri,
                                project=PROJECT_ID, location=REGION)
    model = train_custom_model(gcs_csv_uri=gcs_csv_uri, target_col=target_column,
                               model_gcs_path=model_gcs_path)
    _ = deploy_model(model_gcs_path=model.output, endpoint_display_name=endpoint_display_name)

# -------------------- Main --------------------
if __name__ == '__main__':
    compiler.Compiler().compile(pipeline_func=iris_pipeline, package_path='iris_pipeline.json')
    print('Compiled pipeline to iris_pipeline.json')

    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION)

    job = aiplatform.PipelineJob(display_name='iris-custom-pipeline-run',
                                 template_path='iris_pipeline.json',
                                 pipeline_root=PIPELINE_ROOT)
    job.run(sync=False)
    print('Submitted pipeline job (async).')
