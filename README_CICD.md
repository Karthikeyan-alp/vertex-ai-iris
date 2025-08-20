# How to push repo and create Cloud Build trigger

1. Set gcloud defaults:
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/region us-central1
gcloud services enable cloudbuild.googleapis.com aiplatform.googleapis.com
```

2. Initialize local repo and push to GitHub (example):
```bash
git init
git add .
git commit -m "initial"
git remote add origin git@github.com:YOUR_USER/vertex-ai-iris.git
git push -u origin main
```

3. Create Cloud Build trigger (GitHub):
```bash
gcloud alpha builds triggers create github   --name="vertex-iris-trigger"   --repo-owner="YOUR_USER"   --repo-name="vertex-ai-iris"   --branch-pattern="^main$"   --build-config="cloudbuild.yaml"
```

4. Alternatively, create trigger from Cloud Console: Cloud Build → Triggers → Create Trigger → choose GitHub → select repo → set branch pattern → cloudbuild.yaml.

5. On every push to main, Cloud Build will run and execute `cloudbuild.yaml` steps (data_prep + compile & submit pipeline).

## Where to see experiment tracking
- Vertex AI Console → Experiments → choose experiment name set in .env (EXPERIMENT_NAME).

## Example change to trigger CI/CD
Edit `data_prep.py` to add a new derived column, e.g.:
```python
df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
```
Commit & push → trigger runs.
