import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_time_features(df):

    df["date_time"] = pd.to_datetime(df["date_time"])

    # Hour of day
    df["hour"] = df["date_time"].dt.hour

    # Day of week
    df["day_of_week"] = df["date_time"].dt.day_name()

    # Weekend indicator
    df["is_weekend"] = (
        df["date_time"].dt.dayofweek >= 5
    ).astype(int)

    logger.info(
        "Created hour, day_of_week and is_weekend features."
    )

    return df


def create_cyclical_features(df):

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    logger.info(
        "Created cyclical hour features: hour_sin and hour_cos."
    )

    return df


def encode_weather(df):

    df = pd.get_dummies(
        df,
        columns=["weather_main"],
        prefix="weather",
        dtype=int
    )

    logger.info(
        "Encoded weather_main using one-hot encoding."
    )

    return df


def create_weather_indicator(df):

    adverse_conditions = [
        "Rain",
        "Snow",
        "Thunderstorm",
        "Drizzle",
        "Fog",
        "Mist",
        "Squall"
    ]

    df["adverse_weather"] = (
        df["weather_main"]
        .isin(adverse_conditions)
        .astype(int)
    )

    logger.info(
        "Created adverse_weather indicator."
    )

    return df



def scale_numeric_features(df):

    columns_to_scale = [
        "temp",
        "traffic_volume"
    ]

    for column in columns_to_scale:

        minimum = df[column].min()
        maximum = df[column].max()

        logger.debug(
            "%s scaling values: min=%.2f, max=%.2f",
            column,
            minimum,
            maximum
        )

        if maximum != minimum:

            df[column + "_scaled"] = (
                (df[column] - minimum) /
                (maximum - minimum)
            )

        else:
            df[column + "_scaled"] = 0

            logger.warning(
                "%s could not be normally scaled because "
                "minimum and maximum are equal.",
                column
            )

    logger.info(
        "Created scaled versions of temp and traffic_volume."
    )

    return df



def create_congestion_category(df):

    q1 = df["traffic_volume"].quantile(0.25)
    q3 = df["traffic_volume"].quantile(0.75)

    logger.debug(
        "Traffic volume thresholds: Q1=%.2f, Q3=%.2f",
        q1,
        q3
    )

    conditions = [
        df["traffic_volume"] <= q1,
        df["traffic_volume"] >= q3
    ]

    categories = [
        "Low",
        "High"
    ]

    df["congestion_category"] = np.select(
        conditions,
        categories,
        default="Medium"
    )

    logger.info(
        "Created congestion_category using traffic volume quartiles."
    )

    return df


def engineer_features(df):

    logger.info(
        "Dataset shape before feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )

    df = create_time_features(df)

    df = create_cyclical_features(df)

    df = create_weather_indicator(df)

    df = encode_weather(df)

    df = scale_numeric_features(df)

    df = create_congestion_category(df)

    logger.info(
        "Dataset shape after feature engineering: %d rows, %d columns.",
        df.shape[0],
        df.shape[1]
    )

    return df


def save_feature_data(df, output_path):

    try:
        df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            "Feature-engineered dataset saved to %s.",
            output_path
        )

        return True

    except OSError:
        logger.error(
            "Unable to save feature-engineered dataset.",
            exc_info=True
        )

        return False



