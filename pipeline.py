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
from google.cloud import aiplatform
#added

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'us-central1')
BUCKET_NAME = os.getenv('BUCKET_NAME')
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', f'gs://{BUCKET_NAME}/pipeline_root')
GCS_CSV = os.getenv('DATA_PATH', 'data/iris.csv')
GCS_CSV_URI = f'gs://{BUCKET_NAME}/{GCS_CSV}'

from kfp import dsl, compiler
from kfp.dsl import component

BASE_IMAGE = 'python:3.10'
PKGS = [
    'google-cloud-aiplatform==1.56.0',
    'pandas',
    'scikit-learn',
    'joblib'
]

# -------------------- Dataset Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def ensure_tabular_dataset(project: str, location: str, display_name: str, gcs_uri: str) -> str:
    from google.cloud import aiplatform
    import pandas as pd
    aiplatform.init(project=project, location=location)

    # Check if dataset already exists
    for d in aiplatform.TabularDataset.list():
        if d.display_name == display_name:
            print('Reusing dataset:', d.resource_name)
            return d.resource_name

    # Create dataset
    ds = aiplatform.TabularDataset.create(display_name=display_name, gcs_source=[gcs_uri])
    print('Created dataset:', ds.resource_name)
    return ds.resource_name

# -------------------- Export Dataset Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def export_tabular_to_csv(dataset_resource_name: str) -> str:
    from google.cloud import aiplatform
    import pandas as pd

    aiplatform.init()
    dataset = aiplatform.TabularDataset(dataset_resource_name)
    df = dataset.to_dataframe()
    
    csv_local_path = '/tmp/data.csv'
    df.to_csv(csv_local_path, index=False)
    print('Dataset exported locally at', csv_local_path)
    return csv_local_path

# -------------------- Training Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def train_model(csv_path: str, target_col: str = 'target') -> str:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    model_path = '/tmp/rf_model.joblib'
    joblib.dump(clf, model_path)
    print('Model saved at', model_path)
    return model_path

# -------------------- Deployment Component --------------------
@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def deploy_model_placeholder(model_path: str) -> str:
    print(f'Model ready for deployment at {model_path}')
    return model_path

# -------------------- Pipeline --------------------
@dsl.pipeline(name='iris-tabular-custom-pipeline')
def iris_pipeline(project: str = PROJECT_ID, location: str = REGION,
                  dataset_display_name: str = 'iris-dataset',
                  gcs_csv_uri: str = GCS_CSV_URI,
                  target_column: str = 'target'):

    dataset_res = ensure_tabular_dataset(project=project, location=location,
                                         display_name=dataset_display_name, gcs_uri=gcs_csv_uri)
    
    csv_file = export_tabular_to_csv(dataset_resource_name=dataset_res.output)
    
    model_file = train_model(csv_path=csv_file.output, target_col=target_column)
    
    _ = deploy_model_placeholder(model_path=model_file.output)

# -------------------- Main --------------------
if __name__ == '__main__':
    # Compile pipeline
    compiler.Compiler().compile(pipeline_func=iris_pipeline, package_path='iris_pipeline.json')
    print('Compiled pipeline to iris_pipeline.json')

    # Initialize AI Platform
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)

    # Submit pipeline
    pipeline_job = aiplatform.PipelineJob(
        display_name="iris-tabular-pipeline-job",
        template_path="iris_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project": PROJECT_ID,
            "location": REGION,
            "dataset_display_name": "iris-dataset",
            "gcs_csv_uri": GCS_CSV_URI,
            "target_column": "target"
        },
    )
    pipeline_job.run()
    print("Pipeline submitted successfully!")
