from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

model = joblib.load(
    "models/random_forest_regression.pkl"
)

features = joblib.load(
    "models/regression_features.pkl"
)

logger.info("Traffic prediction model loaded successfully.")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Smart City Traffic Prediction API"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame(
            [data],
            columns=features
        )

        prediction = model.predict(input_df)[0]

        logger.info(
            "Traffic prediction generated successfully."
        )

        return jsonify({
            "predicted_traffic_volume": round(
                float(prediction), 2
            )
        })

    except Exception as e:
        logger.error(
            "Prediction failed.",
            exc_info=True
        )

        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=False,
        port=5001
    )


    