# Smart City Traffic Intelligence --- Part 3

## Machine Learning and AI: Building an Intelligent Mobility Solution

## Project Overview

Part 3 extends the cleaned and feature-engineered traffic dataset
produced in Part 2 into an intelligent mobility solution.

The project applies supervised machine learning, unsupervised learning,
deep learning, explainable AI, MLflow experiment tracking, a traffic
recommendation system, model deployment and monitoring, and responsible
AI analysis.

The main objectives are to classify a documented proxy for accident
risk, predict traffic volume, discover traffic patterns, model
sequential traffic behaviour, explain predictions, recommend
lower-traffic travel periods, and demonstrate MLOps and responsible AI
practices.

## Important Accident Dataset Note

No real accident dataset was supplied for this capstone.

A **proxy accident-risk label** was created for demonstration purposes.
Higher proxy risk is defined using **High or Severe congestion together
with risky weather conditions**.

The classification models demonstrate a machine-learning workflow only
and **must not be interpreted as models that predict actual road
accidents**.

## Project Structure

``` text
capstone_part3/
├── README.md
├── part3_machine_learning.ipynb
├── final_part3_report.pdf
├── bias_fairness_report.pdf
├── app.py
├── monitoring.py
├── recommendation_engine.py
├── data/
│   └── traffic_features.csv
├── models/
│   ├── logistic_classification.pkl
│   ├── random_forest_classification.pkl
│   ├── linear_regression.pkl
│   ├── random_forest_regression.pkl
│   ├── regression_features.pkl
│   └── traffic_lstm.keras
├── outputs/
│   └── travel_recommendations.csv
└── mlruns/
    └── MLflow experiment files
```

Only files that were actually generated should be included in the final
repository.

## Task 1 --- Supervised Machine Learning

### Classification

The classification task predicts the constructed proxy accident-risk
label using Logistic Regression and Random Forest.

  Model                   Accuracy   Precision   Recall   F1-score   ROC AUC
  --------------------- ---------- ----------- -------- ---------- ---------
  Logistic Regression       0.9565      0.8227   0.9871     0.8974    0.9913
  Random Forest             0.9786      0.9089   0.9882     0.9469    0.9969

### Regression

Traffic volume is predicted using Linear Regression and Random Forest
Regression.

  Model                             MAE   R-squared
  -------------------------- ---------- -----------
  Linear Regression            820.3828      0.7220
  Random Forest Regression     266.7202      0.9469

## Task 2 --- Unsupervised Machine Learning

K-means clustering was applied using hour, adverse weather and traffic
volume. The features were standardised before clustering. Four clusters
were retained as an interpretable solution.

    Cluster   Average Hour   Adverse Weather   Average Traffic   Records
  --------- -------------- ----------------- ----------------- ---------
          0          14.37               1.0           4199.78     12096
          1           3.09               0.0            926.38      8127
          2          15.14               0.0           4268.72     21795
          3           3.28               1.0            924.83      6169

Association-rule mining was applied to time of day, weekday/weekend,
weather and congestion. One useful rule was **Weekday + Afternoon →
Severe Congestion**, with support 0.1033, confidence 0.7133 and lift
2.8554. Association rules describe relationships and do not establish
causation.

## Task 3 --- Deep Learning and Explainability

An LSTM was developed for sequential traffic-volume prediction using
24-observation sequences, 50 LSTM units, the Adam optimiser, Mean
Squared Error loss, batch size 32 and early stopping.

  Metric          Result
  ----------- ----------
  MAE           219.0291
  R-squared       0.9742

The LSTM used a sequential evaluation setup, so its metrics should not
be treated as a strictly controlled direct comparison with the
conventional regression models.

SHAP was applied to the comparable Random Forest regression model. The
most influential features were `hour_cos`, `hour`, `day_of_week_num`,
`hour_sin` and `is_weekend`.

## Task 4 --- MLflow Experiment Tracking

MLflow was used to track model experiments under:

``` text
Smart_City_Traffic_Models
```

Tracked models included Logistic Regression Classification, Random
Forest Classification, Linear Regression Traffic Prediction, Random
Forest Regression and LSTM Traffic Prediction.

Classification runs recorded accuracy, precision, recall, F1-score and
ROC AUC. Regression and LSTM runs recorded MAE and R-squared.

## Task 5 --- Traffic Recommendation System

Because the dataset represents one traffic corridor, the recommendation
system focuses on travel timing rather than alternative routes.

  -----------------------------------------------------------------------
  Day Type    Weather         Recommended Average Traffic      Historical
                                     Hour                         Records
  ----------- ----------- --------------- --------------- ---------------
  Weekday     Normal                    2          304.28             831

  Weekday     Adverse                   2          298.93             624

  Weekend     Normal                    4          383.86             334

  Weekend     Adverse                   4          364.62             261
  -----------------------------------------------------------------------

The system recommends approximately **2:00 AM--3:00 AM** for weekday
travel and **4:00 AM--5:00 AM** for weekend travel. Recommendations are
based on historical patterns rather than real-time traffic information.

## Task 6 --- MLOps and Deployment

### Model Versioning

  Version   Model                             MAE   R-squared
  --------- -------------------------- ---------- -----------
  v1        Linear Regression            820.3828      0.7220
  v2        Random Forest Regression     266.7202      0.9469
  v3        LSTM                         219.0291      0.9742

### Flask API

`app.py` provides a deployment simulation for the Random Forest
traffic-volume regression model. A successful test returned:

``` json
{
  "predicted_traffic_volume": 4691.0
}
```

### Monitoring

Normal monitoring:

  Measure                    Result
  ----------------- ---------------
  Baseline MAE               266.72
  Monitoring MAE             252.98
  Alert Threshold            320.06
  Status              PASS / Normal

Simulated deterioration:

  Measure                                     Result
  ----------------- --------------------------------
  Baseline MAE                                266.72
  Drifted MAE                                 508.18
  Alert Threshold                             320.06
  Status              ALERT / Requires investigation

The 20% alert threshold is for demonstration purposes.

## Task 7 --- Responsible and Sustainable AI

### Data Coverage

  Day Type   Weather     Records
  ---------- --------- ---------
  Weekday    Normal        21195
  Weekday    Adverse       13292
  Weekend    Normal         8727
  Weekend    Adverse        4973

### Prediction Error by Condition

  Day Type   Weather        MAE   Test Records
  ---------- --------- -------- --------------
  Weekday    Adverse     263.74           2671
  Weekday    Normal      270.79           4256
  Weekend    Adverse     259.16            971
  Weekend    Normal      265.57           1740

The responsible AI analysis considers unequal data coverage, limitations
of the proxy accident-risk label, human oversight, model governance,
monitoring and computational sustainability. See
`bias_fairness_report.pdf` for the detailed discussion.

## Requirements

Key Python packages used include:

``` text
pandas
numpy
matplotlib
scikit-learn
tensorflow
shap
mlxtend
mlflow
flask
joblib
```

Install missing packages before running the project.

## How to Run the Main Notebook

1.  Start Jupyter Notebook.
2.  Navigate to `capstone_part3`.
3.  Open `part3_machine_learning.ipynb`.
4.  Run the cells in order.

## How to Run the Recommendation Engine

From the `capstone_part3` directory:

``` bash
python recommendation_engine.py
```

The script loads `data/traffic_features.csv` and saves the
recommendation table to `outputs/travel_recommendations.csv`.

## How to Run the Deployment Simulation

From the `capstone_part3` directory:

``` bash
python app.py
```

The Flask application serves the saved Random Forest regression model
through the `/predict` endpoint.

## Model Monitoring

Run:

``` bash
python monitoring.py
```

The monitoring script reports either `PASS / Normal` or
`ALERT / Requires investigation` according to the demonstration error
threshold.

## Logging

Project scripts use:

``` python
logger = logging.getLogger(__name__)
```

`INFO` is used for normal program milestones, `WARNING` for recoverable
but noteworthy conditions, and `ERROR` for failures. `print()` is
reserved for direct end-user output where appropriate.

## Limitations

-   The dataset represents historical observations from a single traffic
    corridor.
-   The accident-risk classification target is a constructed proxy
    rather than actual accident data.
-   Historical traffic patterns may not represent future conditions.
-   The recommendation system does not use real-time incidents,
    construction or road closures.
-   The historically lowest-traffic periods may not always be practical
    travel times.
-   Model performance should be monitored after deployment.
-   Broader data and validation would be required before real-world
    traffic-management or road-safety use.

## Responsible Use

This project demonstrates data science, machine learning and MLOps
techniques. The system should be treated as a **decision-support
demonstration**, not an autonomous traffic-management or
accident-prediction system.

Real-world deployment would require broader data coverage, domain
review, continued monitoring, human oversight and stronger production
governance.
