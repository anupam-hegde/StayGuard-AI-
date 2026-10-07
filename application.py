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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None

    if request.method == "POST":
        try:
            # Construct dictionary of features matching the pipeline schema
            input_dict = {
                "no_of_adults": int(request.form.get("no_of_adults", 1)),
                "no_of_children": int(request.form.get("no_of_children", 0)),
                "no_of_weekend_nights": int(request.form.get("no_of_weekend_nights", 0)),
                "no_of_week_nights": int(request.form.get("no_of_week_nights", 0)),
                "type_of_meal_plan": request.form.get("type_of_meal_plan", "Meal Plan 1"),
                "required_car_parking_space": int(request.form.get("required_car_parking_space", 0)),
                "room_type_reserved": request.form.get("room_type_reserved", "Room_Type 1"),
                "lead_time": float(request.form.get("lead_time", 0.0)),
                "arrival_year": int(request.form.get("arrival_year", 2018)),
                "arrival_month": int(request.form.get("arrival_month", 1)),
                "arrival_date": int(request.form.get("arrival_date", 1)),
                "market_segment_type": request.form.get("market_segment_type", "Online"),
                "repeated_guest": int(request.form.get("repeated_guest", 0)),
                "no_of_previous_cancellations": int(request.form.get("no_of_previous_cancellations", 0)),
                "no_of_previous_bookings_not_canceled": int(request.form.get("no_of_previous_bookings_not_canceled", 0)),
                "avg_price_per_room": float(request.form.get("avg_price_per_room", 0.0)),
                "no_of_special_requests": int(request.form.get("no_of_special_requests", 0)),
            }

            input_df = pd.DataFrame([input_dict])

            if loaded_pipeline is not None:
                raw_pred = loaded_pipeline.predict(input_df)[0]
                prob = float(loaded_pipeline.predict_proba(input_df)[0][1])
                prediction = int(raw_pred)
                logger.info(f"Inference complete: Prediction={prediction}, Probability={prob:.4f}")
            else:
                logger.error("Inference failed: Pipeline artifact not loaded")

        except Exception as e:
            logger.error(f"Error during inference request: {e}")

    return render_template("index.html", prediction=prediction, probability=prob)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
