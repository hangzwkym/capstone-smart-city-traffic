import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def plot_traffic_by_hour(df, output_folder):

    hourly_traffic = (
        df.groupby("hour")["traffic_volume"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_traffic.index,
        hourly_traffic.values,
        marker="o"
    )

    plt.title("Average Traffic Volume by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Volume")
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_path = output_folder / "traffic_by_hour.png"

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved traffic-by-hour figure to %s.",
        file_path
    )


def plot_weekday_weekend(df, output_folder):

    comparison = (
        df.groupby("is_weekend")["traffic_volume"]
        .mean()
    )

    labels = ["Weekday", "Weekend"]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        [
            comparison.get(0, 0),
            comparison.get(1, 0)
        ]
    )

    plt.title("Average Traffic Volume: Weekday vs Weekend")
    plt.xlabel("Day Type")
    plt.ylabel("Average Traffic Volume")
    plt.tight_layout()

    file_path = output_folder / "weekday_weekend.png"

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved weekday-vs-weekend figure to %s.",
        file_path
    )


def plot_temperature_traffic(df, output_folder):

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["temp"],
        df["traffic_volume"],
        alpha=0.3,
        s=10
    )

    plt.title("Temperature vs Traffic Volume")
    plt.xlabel("Temperature (Kelvin)")
    plt.ylabel("Traffic Volume")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_path = output_folder / "temperature_traffic.png"

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Saved temperature-vs-traffic figure to %s.",
        file_path
    )



def create_visualizations(df, output_folder="figures"):

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Starting traffic visualisation generation."
    )

    plot_traffic_by_hour(
        df,
        output_folder
    )

    plot_weekday_weekend(
        df,
        output_folder
    )

    plot_temperature_traffic(
        df,
        output_folder
    )

    logger.info(
        "All traffic visualisations created successfully."
    )





