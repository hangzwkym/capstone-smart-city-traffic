import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_processed_data(file_path):

    try:
        df = pd.read_csv(file_path)

        df["date_time"] = pd.to_datetime(
            df["date_time"]
        )

        logger.info(
            "Processed dataset loaded for CLI application."
        )

        return df

    except FileNotFoundError:
        logger.error(
            "Processed dataset not found: %s",
            file_path
        )
        return None

    except (pd.errors.ParserError, ValueError):
        logger.error(
            "Unable to read processed dataset.",
            exc_info=True
        )
        return None



def query_traffic(df):

    user_input = input(
        "Enter date and time (YYYY-MM-DD HH:MM:SS): "
    )

    logger.info(
        "Command invoked: traffic lookup, argument=%s",
        user_input
    )

    try:
        requested_time = pd.to_datetime(
            user_input,
            format="%Y-%m-%d %H:%M:%S"
        )

    except ValueError:
        logger.error(
            "Invalid date/time entered: %s",
            user_input
        )

        print(
            "Invalid date/time. "
            "Please use YYYY-MM-DD HH:MM:SS."
        )

        return

    result = df[
        df["date_time"] == requested_time
    ]

    if result.empty:

        print(
            "No traffic record was found "
            "for that date and time."
        )

        logger.warning(
            "No traffic record found for %s.",
            requested_time
        )

    else:

        print("\nTraffic information:")

        for _, row in result.iterrows():

            print(
                f"Date/Time: {row['date_time']}"
            )

            print(
                f"Traffic Volume: {row['traffic_volume']}"
            )

            print(
                f"Temperature: {row['temp']:.2f} K"
            )

            print(
                f"Weather: {row['weather_description']}"
            )

            print(
                f"Congestion: {row['congestion_category']}"
            )

            print("-" * 30)



def high_traffic_periods(df):

    logger.info(
        "Command invoked: high traffic periods."
    )

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\nTop 5 High-Traffic Hours:")

    for hour, volume in hourly_traffic.items():

        print(
            f"{int(hour):02d}:00 - "
            f"Average traffic: {volume:.0f}"
        )



def recommended_travel_periods(df):

    logger.info(
        "Command invoked: recommended travel periods."
    )

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
        .sort_values()
        .head(5)
    )

    print(
        "\nRecommended Travel Periods "
        "(Lowest Average Traffic):"
    )

    for hour, volume in hourly_traffic.items():

        print(
            f"{int(hour):02d}:00 - "
            f"Average traffic: {volume:.0f}"
        )


def display_menu():

    print("\n=== Smart City Traffic Analytics ===")

    print("1. Query traffic by date/time")
    print("2. Show high-traffic periods")
    print("3. Show recommended travel periods")
    print("4. Exit")


def main():

    data_file = Path(
        "data/traffic_features.csv"
    )

    df = load_processed_data(
        data_file
    )

    if df is None:
        print(
            "The application could not load "
            "the traffic dataset."
        )
        return

    while True:

        display_menu()

        choice = input(
            "\nEnter your choice (1-4): "
        ).strip()

        logger.info(
            "Menu selection: %s",
            choice
        )

        if choice == "1":

            query_traffic(df)

        elif choice == "2":

            high_traffic_periods(df)

        elif choice == "3":

            recommended_travel_periods(df)

        elif choice == "4":

            logger.info(
                "User exited the traffic application."
            )

            print(
                "Traffic analytics application closed."
            )

            break

        else:

            logger.error(
                "Invalid menu option entered: %s",
                choice
            )

            print(
                "Invalid option. "
                "Please enter 1, 2, 3 or 4."
            )


if __name__ == "__main__":
    main()


def setup_logging():

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        "traffic_app.log",
        mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)



