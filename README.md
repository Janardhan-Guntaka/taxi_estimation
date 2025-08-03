# 🚕 NYC Taxi Ride Prediction

This repository provides all code and configuration to build a **production-grade, end-to-end machine learning system** for forecasting short-term taxi ride demand in New York City. While focused on NYC, the data pipeline and modeling approach can be easily adapted to other cities or similar mobility datasets.

---

## 🎯 Project Goal

Traditional taxi services in urban areas like NYC are sensitive to fast-changing demand. **Accurate short-term forecasts** help operators:
- Optimize fleet allocation and dispatching
- Reduce passenger wait times
- Improve operational efficiency

This project ingests historical taxi trip data, aggregates it into hourly time series, engineers sliding-window features, and trains a LightGBM regressor to **predict number of rides in the next hour**.  
All features and trained models are tracked in [Hopsworks](https://www.hopsworks.ai/) for reproducibility and live inference. An interactive **Streamlit dashboard** visualizes forecasts and location trends.

---

## 🛠️ Pipeline Overview

The repository is structured to support:
- **Data ingestion** (download & clean)
- **Time series aggregation**
- **Feature engineering** (sliding-window, calendar, recent means)
- **Model training** (LightGBM pipeline, MLflow tracking)
- **Automated inference & prediction**
- **Visualization** (Streamlit dashboards)

#### ![Pipeline Overview](assets/pipeline_diagram.png)  
*(Add a pipeline image at the above location for an instant visual overview)*

---

## ⛓️ Key Pipeline Components

| Stage                | Script / Module             | Description                                                |
|----------------------|----------------------------|------------------------------------------------------------|
| **Data Ingestion**   | `src/data_utils.py`        | Download and clean raw trip data, unify schema, remove outliers |
| **Aggregation**      | `src/data_utils.py`        | Aggregate trips to hourly location time series              |
| **Feature Generation** | `src/data_utils.py`        | Generate sliding-window features for each location/hour    |
| **Pipeline/ML**      | `src/pipeline_utils.py`    | Build complete sklearn pipeline; LightGBM, temporal/categorical features |
| **Experiment Tracking** | `src/experiment_utils.py` | MLflow tracking; log models, metrics, register models      |
| **Feature Store**    | `src/feature_pipeline.py`  | Aggregate & push features to Hopsworks feature store       |
| **Training Orchestration** | `pipelines/model_training_pipeline.py` | Full model training pipeline with evaluation & registry    |
| **Real-time Inference** | `pipelines/inference_pipeline.py` | Live hourly prediction pipeline                           |
| **Visualization**    | `frontend/`                | Streamlit-based dashboards for predictions & monitoring    |

---

## ⚙️ GitHub Actions Automation

| Workflow File                              | Schedule                    | Description                                        |
|--------------------------------------------|-----------------------------|----------------------------------------------------|
| `.github/workflows/feature_pipeline.yaml`  | Hourly on the hour          | Runs feature aggregation & loads to Hopsworks      |
| `.github/workflows/inference_pipeline.yaml`| Hourly, 5 min after the hour| Runs live inference, saves fresh predictions       |
| `.github/workflows/model_training_pipeline.yaml` | Weekly (Mon, 00:00)    | Retrains & registers best new model                |

**These workflows ensure continuous data ingestion, up-to-date predictions, and regular model retraining—fully automated.**

---

## 🚀 Getting Started

### 1. **Requirements**

- **Python 3.10**
- Packages listed in `requirements.txt`:
    - pandas, numpy, scikit-learn, lightgbm, hopsworks, mlflow, streamlit, folium, geopandas, shapely, plotly

For deterministic installs: use `requirements_with_version.txt`

### 2. **Installation**

git clone https://github.com/YourUsername/nyc_taxi_prediction.git
cd nyc_taxi_prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

text

Also sign up on [Hopsworks](https://www.hopsworks.ai/), create a project, and set environment variables:

export HOPSWORKS_API_KEY=<your_api_key>
export HOPSWORKS_PROJECT_NAME=<your_project_name>

text

---

### 3. **Running the Pipelines Locally**

**Ingest & clean data:**

python -c "from src.data_utils import load_and_process_taxi_data; df = load_and_process_taxi_data(2024, ); print(df.head())"

text

**Insert features to Hopsworks:**

python -m pipelines.feature_pipeline

text

**Train a new model:**

python -m pipelines.model_training_pipeline

text

**Run inference for next-hour prediction:**

python -m pipelines.inference_pipeline

text

*All scripts authenticate using your saved Hopsworks credentials.*

---

### 4. **Launching the Streamlit Dashboard**

*To explore predictions interactively:*

streamlit run frontend/frontend_v2.py

text
- This will open a browser map of NYC locations, colored by predicted demand. Click a location to see historical and forecasted rides.

*For monitoring and error tracking:*

streamlit run frontend/frontend_monitor.py

text

---

## 🗃️ Data Sources

All raw trip data is public and downloaded from NYC taxi & limousine commission’s official datasets.  
Data is cleaned (e.g., outlier removal, schema unification) and aggregated into complete hourly time series as part of the pipeline.

---

## 🤝 Contributing

Contributions are welcome! To add new features, support more cities, or improve performance:

1. Fork and create a new feature branch.
2. Add tests as needed and ensure existing tests pass.
3. Update documentation (e.g., README).
4. Open a pull request describing your change.

Please:
- Adhere to clean, modular code with comments.
- Open issues and requests in GitHub’s issue tracker.

---

## ⚖️ License

This project is licensed under the **MIT License** (see LICENSE file).

---

## 📢 Acknowledgements

- Built with support from the open-source community, NYC TLC, Hopsworks, and contributors to Python’s scientific stack.
- Special thanks to [Hopsworks](https://www.hopsworks.ai/) for providing the feature store and model registry.
