import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'us-central1')
BUCKET_NAME = os.getenv('BUCKET_NAME')
PIPELINE_ROOT = os.getenv('PIPELINE_ROOT', f'gs://{BUCKET_NAME}/pipeline_root')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'iris-experiment')
GCS_CSV = os.getenv('DATA_PATH', 'data/iris.csv')
GCS_CSV_URI = f'gs://{BUCKET_NAME}/{GCS_CSV}'

from kfp import dsl, compiler
from kfp.dsl import component

BASE_IMAGE = 'python:3.10'
PKGS = ['google-cloud-aiplatform==1.56.0']

@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def ensure_tabular_dataset(project: str, location: str, display_name: str, gcs_uri: str) -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    # reuse if exists
    for d in aiplatform.TabularDataset.list():
        if d.display_name == display_name:
            print('Reusing dataset', d.resource_name)
            return d.resource_name
    ds = aiplatform.TabularDataset.create(display_name=display_name, gcs_source=[gcs_uri])
    print('Created dataset', ds.resource_name)
    return ds.resource_name

@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def train_automl(project: str, location: str, experiment: str, dataset_resource_name: str, target_col: str, model_display_name: str, budget_mnh: int) -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location, experiment=experiment)
    ds = aiplatform.TabularDataset(dataset_resource_name)
    aiplatform.log_params({'target_column': target_col, 'budget_mnh': budget_mnh})
    job = aiplatform.AutoMLTabularTrainingJob(display_name=model_display_name + '-job', optimization_prediction_type='classification', optimization_objective='maximize-log-likelihood')
    model = job.run(dataset=ds, target_column=target_col, model_display_name=model_display_name, budget_milli_node_hours=budget_mnh, sync=True)
    try:
        metrics = model.evaluate()
        if isinstance(metrics, dict):
            aiplatform.log_metrics(metrics)
        else:
            aiplatform.log_metrics({'eval': str(metrics)})
    except Exception as e:
        print('Eval logging skipped:', e)
    return model.resource_name

@component(base_image=BASE_IMAGE, packages_to_install=PKGS)
def deploy_model(project: str, location: str, model_resource_name: str, endpoint_display_name: str, machine_type: str = 'n1-standard-2') -> str:
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)
    model = aiplatform.Model(model_resource_name)
    endpoint = model.deploy(deployed_model_display_name=endpoint_display_name, machine_type=machine_type, min_replica_count=1, max_replica_count=1)
    print('Deployed endpoint', endpoint.resource_name)
    return endpoint.resource_name

@dsl.pipeline(name='iris-automl-pipeline')
def iris_pipeline(project: str = PROJECT_ID, location: str = REGION, experiment: str = EXPERIMENT_NAME, dataset_display_name: str = 'iris-dataset', gcs_csv_uri: str = GCS_CSV_URI, target_column: str = 'target', model_display_name: str = 'iris-automl-model', endpoint_display_name: str = 'iris-endpoint', budget_mnh: int = 300):
    ds = ensure_tabular_dataset(project=project, location=location, display_name=dataset_display_name, gcs_uri=gcs_csv_uri)
    model = train_automl(project=project, location=location, experiment=experiment, dataset_resource_name=ds.output, target_col=target_column, model_display_name=model_display_name, budget_mnh=budget_mnh)
    _ = deploy_model(project=project, location=location, model_resource_name=model.output, endpoint_display_name=endpoint_display_name)

if __name__ == '__main__':
    # compile
    compiler.Compiler().compile(pipeline_func=iris_pipeline, package_path='iris_pipeline.json')
    print('Compiled pipeline to iris_pipeline.json')
    # submit
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME)
    job = aiplatform.PipelineJob(display_name='iris-automl-pipeline-run', template_path='iris_pipeline.json', pipeline_root=PIPELINE_ROOT, parameter_values={'project': PROJECT_ID, 'location': REGION, 'experiment': EXPERIMENT_NAME})
    job.run(sync=False)
    print('Submitted pipeline job (async).')