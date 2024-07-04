# Hospital Stay Forecasting System

![Hospital Bed Management System](https://blogimages.softwaresuggest.com/blog/wp-content/uploads/2019/04/13185138/10-Most-Popular-Hospital-Bed-Management-System.jpg)


## Project Description

### Business Overview

One important area of healthcare analytics is the analysis of patient length of stay (LOS), which refers to the amount of time a patient spends in a healthcare facility. Length of stay is a critical metric in healthcare, as it can impact patient outcomes, healthcare costs, and hospital capacity. By analyzing patient LOS data, healthcare providers can identify opportunities to improve the delivery of care and reduce costs.
In addition to improving patient outcomes, the analysis of patient LOS can also help healthcare providers to reduce costs. For example, by identifying patients who are at risk of extended LOS, providers can take proactive steps to ensure they receive the care they need in a timely manner.


### Aim

To build an end-to-end model that predicts the length of stay for patients being admitted  and implement a retraining pipeline that checks for  data/model drift and redeploys the model if needed.

## Data
- **Training Data**: Present in Snowflake for about 230k patients across various regions and hospitals. A total of 19 features are available in the data.
- **Simulation Data**: Available for 71k patients for prediction purposes.

## Architecture

![Hospital Bed Management System](https://projex.gumlet.io/aws-sagemaker-healthcare-analytics-project/images/image_97475610451686029487776.png?w=1080&dpr=1.3)


## Key Features

- **Training Phase**: Historical patient data stored in a dynamic Snowflake database.
- **Exploratory Data Analysis (EDA)**: Performed to extract relevant and useful information about features determining the Length of Stay.
- **Model Development**: Utilized Amazon SageMaker Notebook instance to build the model, with feature selection and feature engineering techniques to identify and build important features.
- **Inference Pipeline**: 
  - AWS SageMaker instance runs 24/7.
  - The inference pipeline gets triggered at a scheduled time each day 
  - Pulls daily new incoming patient data from Snowflake.
  - Predicts length of stay for new patients.
  - Updates the Snowflake database with predictions.
- **Retraining Pipelines**:
  - Periodically checks for model and data drift using the Alibi package.
  - Retrains the model and compares its performance against the old model.
  - Deploys the new model if it outperforms the existing one.





## Tech Stack

- **Tools**: `AWS SageMaker`, `Snowflake`, `Jupyter Notebook`
- **Language**: `Python`
- **Libraries**: `snowflake-connector-python`, `snowflake-sqlalchemy`, `xgboost`, `pandas`, `numpy`, `scikit-learn`, `alibi`, `scipy`

## Project Structure

```plaintext
.
├── archive
│   ├── old_model_features.pkl
│   ├── old_model_metrics.pkl
│   └── old_model.model
├── artifacts
│   ├── drift_detector.pkl
│   ├── final_features.pkl
│   ├── model_ref_metrics.pkl
│   ├── training_data_with_final_features.pkl
│   └── xgb.model
├── retraining_artifacts
│   ├── retrained_final_features.pkl
│   ├── retrained_model_metrics.pkl
│   └── xgb_retrained.model
├── .gitignore
├── model_building.ipynb
├── model_deployment.ipynb
├── model_monitoring.ipynb
├── model_retraining.ipynb
├── pipeline.ipynb
├── utils.py
├── requirements.txt
└── README.md
```
The project is organized into several components:

- `archive`: Directory that contains the artifacts of old model which has become obsolete due to data/model drift
    - `old_model_features.pkl`: Features used by the old model.
    - `old_model_metrics.pkl`: Performance metrics of the old model.
    - `old_model.model`: Serialized old model.

- `artifacts`: Directory that contains the artifacts of the currently deployed model which runs in production.
    - `final_features.pkl`: Features used by the deployed model.
    - `model_ref_metrics.pkl`: Performance metrics of the deployed model.
    - `xgb.model`: Serialized deployed model.
    - `training_data_with_final_features` : Serialized form of training data with selected final features
    - `drift_detector`: Drift detector that detects the data drift.

- `retraining_artifacts`: Directory that contains the artifacts of the currently deployed model which runs in production.
    - `retrained_final_features.pkl`: Features used by the retrained model.
    - `retrained_model_metrics.pkl`: Performance metrics of the retrained model.
    - `old_model.model`: Serialized retrained model.

- `model_building.ipynb`: Python script for building the model.
- `model_deployment.ipynb`: Python script for deploying the trained model.
- `model_monitoring.ipynb`: Python script for monitoring data or model drift.
- `model_retraining.ipynb`: Python script for retraining the model due to drift.
- `pipeline`: Scheduled notebook for continuous deployment, monitoring, and retraining.
- `utils.py`: Python file contains utility/helper functions.
