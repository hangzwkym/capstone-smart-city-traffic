import logging
from pathlib import Path
from datetime import datetime

import pandas as pd


# --------------------------------------------------
# Logging setup
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# File paths
# --------------------------------------------------

DATA_PATH = Path("data/traffic_features.csv")
OUTPUT_PATH = Path("outputs/travel_recommendations.csv")


# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------

def load_data():
    """Load and prepare the traffic dataset."""

    logger.info("Loading traffic dataset.")

    df = pd.read_csv(DATA_PATH)

    df["date_time"] = pd.to_datetime(df["date_time"])

    # Create readable day type
    df["day_type"] = df["is_weekend"].map({
        0: "Weekday",
        1: "Weekend"
    })

    # Create readable weather type
    df["weather_type"] = df["adverse_weather"].map({
        0: "Normal",
        1: "Adverse"
    })

    logger.info("Traffic dataset loaded successfully.")

    return df


# --------------------------------------------------
# Calculate historical traffic patterns
# --------------------------------------------------

def create_travel_patterns(df):
    """Calculate average traffic by day type, weather and hour."""

    logger.info("Calculating historical travel patterns.")

    travel_patterns = (
        df.groupby(
            ["day_type", "weather_type", "hour"]
        )["traffic_volume"]
        .agg(["mean", "count"])
        .reset_index()
    )

    travel_patterns.columns = [
        "day_type",
        "weather_type",
        "hour",
        "avg_traffic",
        "records"
    ]

    return travel_patterns


# --------------------------------------------------
# Find recommended travel time
# --------------------------------------------------

def recommend_travel_time(
    travel_patterns,
    day_type,
    weather_type,
    minimum_records=20
):
    """Find the lowest-traffic hour for the selected conditions."""

    filtered = travel_patterns[
        (travel_patterns["day_type"] == day_type)
        & (travel_patterns["weather_type"] == weather_type)
    ].copy()

    # Avoid recommendations based on very small samples
    filtered = filtered[
        filtered["records"] >= minimum_records
    ]

    if filtered.empty:
        logger.warning(
            "Insufficient historical data for %s / %s.",
            day_type,
            weather_type
        )
        return None

    best = filtered.sort_values(
        "avg_traffic"
    ).iloc[0]

    return {
        "day_type": day_type,
        "weather_type": weather_type,
        "hour": int(best["hour"]),
        "avg_traffic": round(best["avg_traffic"], 2),
        "records": int(best["records"])
    }


# --------------------------------------------------
# Format hour
# --------------------------------------------------

def format_hour(hour):
    """Convert a 24-hour integer into readable time."""

    return datetime.strptime(
        str(hour),
        "%H"
    ).strftime("%I:%M %p")


# --------------------------------------------------
# Generate plain-language recommendation
# --------------------------------------------------

def generate_recommendation(
    travel_patterns,
    day_type,
    weather_type
):
    """Generate a commuter-friendly recommendation."""

    result = recommend_travel_time(
        travel_patterns,
        day_type,
        weather_type
    )

    if result is None:
        return (
            "There is not enough historical data "
            "to make a recommendation."
        )

    start_hour = result["hour"]
    end_hour = (start_hour + 1) % 24

    start_text = format_hour(start_hour)
    end_text = format_hour(end_hour)

    return (
        f"For a {day_type.lower()} journey during "
        f"{weather_type.lower()} weather, consider travelling "
        f"between {start_text} and {end_text}. "
        f"Historical traffic volume during this period averaged "
        f"approximately {result['avg_traffic']:.0f} vehicles."
    )


# --------------------------------------------------
# Generate recommendation table
# --------------------------------------------------

def create_recommendation_table(travel_patterns):
    """Generate recommendations for all four scenarios."""

    scenarios = [
        ("Weekday", "Normal"),
        ("Weekday", "Adverse"),
        ("Weekend", "Normal"),
        ("Weekend", "Adverse")
    ]

    recommendation_results = []

    for day_type, weather_type in scenarios:

        result = recommend_travel_time(
            travel_patterns,
            day_type,
            weather_type
        )

        if result is not None:

            recommendation_results.append({
                "Day Type": day_type,
                "Weather": weather_type,
                "Recommended Hour": result["hour"],
                "Average Traffic": result["avg_traffic"],
                "Historical Records": result["records"]
            })

    return pd.DataFrame(recommendation_results)


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main():

    df = load_data()

    travel_patterns = create_travel_patterns(df)

    recommendation_table = create_recommendation_table(
        travel_patterns
    )

    # Create outputs folder if necessary
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    recommendation_table.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.info(
        "Travel recommendations saved to %s.",
        OUTPUT_PATH
    )

    # Direct end-user output
    print("\nSMART CITY TRAVEL RECOMMENDATIONS\n")

    scenarios = [
        ("Weekday", "Normal"),
        ("Weekday", "Adverse"),
        ("Weekend", "Normal"),
        ("Weekend", "Adverse")
    ]

    for day_type, weather_type in scenarios:

        recommendation = generate_recommendation(
            travel_patterns,
            day_type,
            weather_type
        )

        print(recommendation)
        print()


if __name__ == "__main__":
    main()