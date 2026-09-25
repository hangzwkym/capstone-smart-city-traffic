import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

from feature_engineering import engineer_features, save_feature_data
from visualizations import create_visualizations

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_logging():

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicates in Jupyter
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        "pipeline.log",
        mode="w"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        logger.info(
            "Raw dataset loaded successfully: %d rows, %d columns",
            df.shape[0],
            df.shape[1]
        )

        return df

    except FileNotFoundError:
        logger.error(
            "CSV file was not found: %s",
            file_path,
            exc_info=True
        )
        return None

    except pd.errors.ParserError:
        logger.error(
            "CSV file could not be parsed: %s",
            file_path,
            exc_info=True
        )
        return None

    except OSError:
        logger.error(
            "File I/O error while loading: %s",
            file_path,
            exc_info=True
        )
        return None



EXPECTED_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
    "traffic_volume"
]


def validate_schema(df):
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            "Schema validation failed. Missing columns: %s",
            missing_columns
        )
        return False

    logger.info("Schema validation successful.")
    return True





def clean_holiday(df):
    missing_count = df["holiday"].isna().sum()

    if missing_count > 0:
        df["holiday"] = df["holiday"].fillna("None")

        logger.warning(
            "%d missing holiday values replaced with 'None'.",
            missing_count
        )
    else:
        logger.info("No missing holiday values found.")

    return df


def standardise_categories(df):
    categorical_columns = [
        "holiday",
        "weather_main",
        "weather_description"
    ]

    for column in categorical_columns:
        before = df[column].copy()

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

        changed = (before.astype(str) != df[column]).sum()

        if changed > 0:
            logger.warning(
                "%d values standardised in column '%s'.",
                changed,
                column
            )
        else:
            logger.info(
                "No inconsistent spacing found in column '%s'.",
                column
            )

    return df



def clean_datetime(df):
    df["date_time"] = pd.to_datetime(
        df["date_time"],
        errors="coerce"
    )

    invalid_dates = df["date_time"].isna().sum()

    if invalid_dates > 0:
        df = df.dropna(subset=["date_time"])

        logger.warning(
            "%d rows removed because date_time could not be parsed.",
            invalid_dates
        )
    else:
        logger.info("All date_time values parsed successfully.")

    return df



def remove_duplicates(df):
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

        logger.warning(
            "%d duplicate rows removed.",
            duplicate_count
        )
    else:
        logger.info("No duplicate rows found.")

    return df


def clean_temperature(df):
    invalid_count = (df["temp"] <= 0).sum()

    if invalid_count > 0:

        df["month"] = df["date_time"].dt.month

        for month in range(1, 13):

            valid_month_values = df.loc[
                (df["month"] == month) &
                (df["temp"] > 0),
                "temp"
            ]

            if not valid_month_values.empty:

                monthly_median = valid_month_values.median()

                mask = (
                    (df["month"] == month) &
                    (df["temp"] <= 0)
                )

                affected = mask.sum()

                if affected > 0:
                    df.loc[mask, "temp"] = monthly_median

                    logger.warning(
                        "%d invalid temperature values imputed "
                        "using month %d median %.2f K.",
                        affected,
                        month,
                        monthly_median
                    )

        df = df.drop(columns=["month"])

    else:
        logger.info("No invalid temperature values found.")

    return df



def clean_rainfall(df):
    invalid_mask = (df["rain_1h"] < 0) | (df["rain_1h"] > 9000)
    invalid_count = invalid_mask.sum()

    if invalid_count > 0:
        valid_median = df.loc[
            ~invalid_mask,
            "rain_1h"
        ].median()

        df.loc[invalid_mask, "rain_1h"] = valid_median

        logger.warning(
            "%d invalid rainfall values imputed with median %.2f mm.",
            invalid_count,
            valid_median
        )
    else:
        logger.info("No invalid rainfall values found.")

    return df



def validate_numeric_values(df):

    negative_snow = (df["snow_1h"] < 0).sum()

    if negative_snow > 0:
        df.loc[df["snow_1h"] < 0, "snow_1h"] = 0

        logger.warning(
            "%d negative snow values replaced with 0.",
            negative_snow
        )
    else:
        logger.info("No negative snow values found.")

    invalid_clouds = (
        (df["clouds_all"] < 0) |
        (df["clouds_all"] > 100)
    )

    cloud_count = invalid_clouds.sum()

    if cloud_count > 0:
        median_clouds = df.loc[
            ~invalid_clouds,
            "clouds_all"
        ].median()

        df.loc[invalid_clouds, "clouds_all"] = median_clouds

        logger.warning(
            "%d invalid cloud coverage values imputed with median.",
            cloud_count
        )
    else:
        logger.info("No invalid cloud coverage values found.")

    return df



def save_cleaned_data(df, output_path):
    try:
        df.to_csv(output_path, index=False)

        logger.info(
            "Cleaned dataset saved successfully to %s",
            output_path
        )

    except OSError:
        logger.error(
            "Unable to save cleaned dataset to %s",
            output_path,
            exc_info=True
        )
        return False

    return True



def main():

    setup_logging()

    input_file = Path(
        "data/Metro_Interstate_Traffic_Volume.csv"
    )

    output_file = Path(
        "data/traffic_cleaned.csv"
    )

    logger.info("Traffic data pipeline started.")

    df = load_data(input_file)

    if df is None:
        logger.error("Pipeline stopped because data could not be loaded.")
        return

    # Schema must be validated before cleaning
    if not validate_schema(df):
        logger.error("Pipeline stopped because schema validation failed.")
        return

    df = clean_holiday(df)

    df = standardise_categories(df)

    df = clean_datetime(df)

    df = remove_duplicates(df)

    df = clean_temperature(df)

    df = clean_rainfall(df)

    df = validate_numeric_values(df)

    success = save_cleaned_data(
        df,
        output_file
    )


    logger.info("Starting feature engineering.")

    df = engineer_features(df)
    
    feature_output = Path(
        "data/traffic_features.csv"
    )
    
    feature_success = save_feature_data(
        df,
        feature_output
    )
    
    if not feature_success:
        logger.error(
            "Pipeline stopped because feature data could not be saved."
        )
        return

    logger.info(
        "Starting visualisation stage."
    )
    
    create_visualizations(
        df,
        "figures"
    )
    
    if success:
        logger.info(
            "Traffic data pipeline completed successfully. "
            "Final dataset: %d rows, %d columns.",
            df.shape[0],
            df.shape[1]
        )
    else:
        logger.error("Pipeline could not complete successfully.")


if __name__ == "__main__":
    main()


