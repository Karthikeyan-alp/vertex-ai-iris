# Vertex AI Iris AutoML - CI/CD example (Cloud Build)

This repo contains a minimal, low-cost end-to-end example to run the Iris dataset on Vertex AI with:
- data prep (download Iris & upload to GCS)
- AutoML Tabular training inside a Vertex AI Pipeline
- Experiment tracking (Vertex AI Experiments)
- Deployment to a small endpoint
- CI/CD via Cloud Build (trigger on push)
- Cleanup script to delete models/endpoints to save cost

## Quick steps (local test)
1. Copy `.env.sample` to `.env` and fill values.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data (uploads iris.csv to GCS):
   ```bash
   python data_prep.py
   ```
4. Compile and submit pipeline (will run in Vertex AI):
   ```bash
   python pipeline.py
   ```
5. After testing, delete resources:
   ```bash
   python cleanup.py
   ```

## CI/CD with Cloud Build
1. Push this repo to GitHub or Cloud Source Repositories.
2. In GCP Console: Cloud Build → Triggers → Create Trigger
   - Source: connect your GitHub repo (or use Cloud Source Repos)
   - Event: Push to branch (e.g., main)
   - Build config: cloudbuild.yaml
3. On every push, Cloud Build will run and trigger the pipeline.

## Where to view experiment runs
- Vertex AI → Experiments → select `EXPERIMENT_NAME` from `.env`

