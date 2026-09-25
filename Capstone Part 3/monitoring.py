import logging
import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


def check_prediction_error(
    actual,
    predicted,
    baseline_mae,
    threshold_percent=0.20
):
    current_mae = mean_absolute_error(
        actual,
        predicted
    )

    threshold = baseline_mae * (
        1 + threshold_percent
    )

    if current_mae > threshold:
        status = "ALERT / Requires investigation"

        logger.warning(
            "Prediction error monitoring alert triggered. "
            "Current MAE: %.2f, threshold: %.2f",
            current_mae,
            threshold
        )

    else:
        status = "PASS / Normal"

        logger.info(
            "Prediction error monitoring passed. "
            "Current MAE: %.2f, threshold: %.2f",
            current_mae,
            threshold
        )

    return {
        "current_mae": current_mae,
        "threshold": threshold,
        "status": status
    }

