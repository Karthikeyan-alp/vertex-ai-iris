import os
from dotenv import load_dotenv
load_dotenv()
from google.cloud import aiplatform

PROJECT_ID = os.getenv('PROJECT_ID')
REGION = os.getenv('REGION', 'us-central1')

aiplatform.init(project=PROJECT_ID, location=REGION)

def cleanup():
    print('Deleting endpoints...')
    for ep in aiplatform.Endpoint.list():
        try:
            print('Undeploying and deleting', ep.display_name)
            ep.undeploy_all()
            ep.delete()
        except Exception as e:
            print('Endpoint delete error:', e)
    print('Deleting models...')
    for m in aiplatform.Model.list():
        try:
            print('Deleting model', m.display_name)
            m.delete()
        except Exception as e:
            print('Model delete error:', e)
    print('Cleanup complete.')

if __name__ == '__main__':
    cleanup()