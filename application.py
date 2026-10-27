import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from config.paths_config import MODEL_OUTPUT_PATH
from src.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__)

# Load unified pipeline artifact
if os.path.exists(MODEL_OUTPUT_PATH):
    loaded_pipeline = joblib.load(MODEL_OUTPUT_PATH)
    logger.info(f"Loaded unified pipeline artifact from {MODEL_OUTPUT_PATH}")
else:
    loaded_pipeline = None
    logger.warning(f"Model artifact not found at {MODEL_OUTPUT_PATH}. Run training pipeline first.")

def compute_risk_profile(prob: float, pred: int):
    cancellation_pct = round(prob * 100, 1)
    retention_pct = round((1.0 - prob) * 100, 1)
    if prob >= 0.70:
        risk_level = "High"
        badge_color = "#ef4444"
        action = "High Risk of Cancellation. Recommend requiring card pre-authorization, deposit, or offering an immediate non-refundable rate discount."
    elif prob >= 0.40:
        risk_level = "Medium"
        badge_color = "#f59e0b"
        action = "Moderate Risk. Send an automated pre-arrival concierge email and confirm guest itinerary 48 hours prior."
    else:
        risk_level = "Low"
        badge_color = "#10b981"
        action = "Low Risk / High Confidence. Standard reservation workflow with priority check-in readiness."

    return {
        "prediction": pred,
        "prediction_label": "Canceled" if pred == 1 else "Not_Canceled",
        "cancellation_probability": prob,
        "cancellation_pct": cancellation_pct,
        "retention_pct": retention_pct,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "action": action,
    }


def parse_features_from_dict(data: dict) -> pd.DataFrame:
    features = {
        "no_of_adults": int(data.get("no_of_adults", 1)),
        "no_of_children": int(data.get("no_of_children", 0)),
        "no_of_weekend_nights": int(data.get("no_of_weekend_nights", 0)),
        "no_of_week_nights": int(data.get("no_of_week_nights", 0)),
        "type_of_meal_plan": str(data.get("type_of_meal_plan", "Meal Plan 1")),
        "required_car_parking_space": int(data.get("required_car_parking_space", 0)),
        "room_type_reserved": str(data.get("room_type_reserved", "Room_Type 1")),
        "lead_time": float(data.get("lead_time", 0.0)),
        "arrival_year": int(data.get("arrival_year", 2018)),
        "arrival_month": int(data.get("arrival_month", 10)),
        "arrival_date": int(data.get("arrival_date", 15)),
        "market_segment_type": str(data.get("market_segment_type", "Online")),
        "repeated_guest": int(data.get("repeated_guest", 0)),
        "no_of_previous_cancellations": int(data.get("no_of_previous_cancellations", 0)),
        "no_of_previous_bookings_not_canceled": int(data.get("no_of_previous_bookings_not_canceled", 0)),
        "avg_price_per_room": float(data.get("avg_price_per_room", 100.0)),
        "no_of_special_requests": int(data.get("no_of_special_requests", 0)),
    }
    return pd.DataFrame([features])


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None
    risk_info = None

    if request.method == "POST":
        try:
            input_df = parse_features_from_dict(request.form)

            if loaded_pipeline is not None:
                raw_pred = int(loaded_pipeline.predict(input_df)[0])
                prob = float(loaded_pipeline.predict_proba(input_df)[0][1])
                prediction = raw_pred
                risk_info = compute_risk_profile(prob, prediction)
                logger.info(f"Inference complete: Prediction={prediction}, Probability={prob:.4f}")
            else:
                logger.error("Inference failed: Pipeline artifact not loaded")

        except Exception as e:
            logger.error(f"Error during inference request: {e}")

    return render_template("index.html", prediction=prediction, probability=prob, risk_info=risk_info)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if loaded_pipeline is None:
        return {"error": "Pipeline artifact not loaded on server"}, 503

    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        input_df = parse_features_from_dict(data)

        raw_pred = int(loaded_pipeline.predict(input_df)[0])
        prob = float(loaded_pipeline.predict_proba(input_df)[0][1])
        result = compute_risk_profile(prob, raw_pred)

        logger.info(f"AJAX Inference: Pred={raw_pred}, Prob={prob:.4f}, Risk={result['risk_level']}")
        return result, 200

    except Exception as e:
        logger.error(f"API Inference failed: {e}")
        return {"error": str(e)}, 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
