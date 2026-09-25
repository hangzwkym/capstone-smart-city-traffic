# Smart City Traffic Analytics Pipeline

## Project Overview

This project develops a reproducible Python traffic analytics pipeline using the Metro Interstate Traffic Volume dataset. The project supports a Smart City Mobility Analytics Team by cleaning traffic data, engineering useful features, generating traffic visualisations, and providing a command-line application for traffic analysis.

## Project Structure

capstone_part2/
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv
│   ├── traffic_cleaned.csv
│   └── traffic_features.csv
│
├── figures/
│   ├── traffic_by_hour.png
│   ├── weekday_weekend.png
│   └── temperature_traffic.png
│
├── pipeline.py
├── feature_engineering.py
├── visualizations.py
├── traffic_app.py
├── pipeline.log
├── traffic_app.log
└── README.md

## Data Pipeline

The main pipeline is implemented in `pipeline.py`.

The pipeline performs the following steps:

1. Loads the raw traffic dataset.
2. Validates the required schema.
3. Handles missing holiday values.
4. Standardises categorical values.
5. Parses and validates date/time values.
6. Removes exact duplicate rows.
7. Detects and handles invalid temperature, rainfall, snow and cloud values.
8. Saves the cleaned dataset.
9. Performs feature engineering.
10. Generates and saves traffic visualisations.

## Feature Engineering

The `feature_engineering.py` module creates additional variables for analysis and machine learning.

Features include:

- Hour of day
- Day of week
- Weekend indicator
- Cyclical hour encoding using sine and cosine
- Adverse weather indicator
- One-hot encoded weather variables
- Scaled temperature
- Scaled traffic volume
- Congestion category

The congestion category is created using traffic volume quartiles. Values at or below the first quartile are classified as Low, values at or above the third quartile are classified as High, and the remaining values are classified as Medium.

## Visualisations

The `visualizations.py` module creates three Matplotlib visualisations:

1. Average traffic volume by hour.
2. Average weekday versus weekend traffic.
3. Relationship between temperature and traffic volume.

The generated figures are saved in the `figures` folder.

## Command-Line Application

The `traffic_app.py` file provides an interactive traffic analytics application.

The application supports three main functions:

1. Query traffic information for a specific date and time.
2. Display the five highest-traffic hours based on historical average traffic volume.
3. Recommend travel periods based on the five hours with the lowest historical average traffic volume.

The application also validates user input and displays clear error messages when invalid input is entered.

## How to Run the Pipeline

Open Anaconda Prompt and navigate to the project directory.

Run:

python pipeline.py

The pipeline will clean the data, perform feature engineering, save the processed datasets and generate the visualisations.

The pipeline can also be executed from Jupyter Notebook using:

%run pipeline.py

## How to Run the Traffic Application

From Anaconda Prompt, run:

python traffic_app.py

Alternatively, from Jupyter Notebook run:

%run traffic_app.py

Follow the menu instructions and enter a number from 1 to 4.

## Logging

Python's `logging` module is used throughout the project.

The logging configuration records:

- INFO messages for successful processing steps.
- WARNING messages when data is modified, removed or imputed.
- ERROR messages when the pipeline or application encounters a problem.
- DEBUG messages for detailed intermediate calculations when DEBUG mode is enabled.

Log messages contain the timestamp, log level, module name and message.

Pipeline logs are stored in `pipeline.log`. The command-line application also records user commands and invalid inputs in its log file.

## Requirements

The project uses:

- Python
- Pandas
- NumPy
- Matplotlib

## Reproducibility

The project separates data cleaning, feature engineering, visualisation and user interaction into individual Python modules. Git is used for version control, with incremental commits documenting major stages of development.