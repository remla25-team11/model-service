# In model-service/service/app.py

import os
import json
import requests
import joblib
import logging
from flask import Flask, request, jsonify, Response
from flasgger import Swagger
from flask_cors import CORS
# Assuming lib_ml is installed from your requirements.txt
from lib_ml.preprocessor import preprocess_text

# Prometheus client
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# --- THIS IS THE CRITICAL SECTION TO FIX ---

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)

# Get model paths from environment variables set by Kubernetes/Helm.
# These paths point to files *inside* the Docker image.
# There are no default fallbacks, so the app will fail fast if the env vars are missing.
MODEL_PATH = os.getenv("MODEL_PATH")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "unknown")

# Check if the environment variables are set
if not MODEL_PATH or not VECTORIZER_PATH:
    logging.error("FATAL: MODEL_PATH or VECTORIZER_PATH environment variables not set.")
    # Exit with a non-zero code to make the container crash clearly if not configured.
    exit(1)

# Load the models at startup, these files should already be in the image.
try:
    logging.info(f"Attempting to load model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")

    logging.info(f"Attempting to load vectorizer from: {VECTORIZER_PATH}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("Vectorizer loaded successfully.")

except FileNotFoundError as e:
    logging.error(f"FATAL: Error loading model files: {e}")
    logging.error("Please ensure the model files were correctly downloaded/copied in the Dockerfile.")
    # Exit with a non-zero code
    exit(1)
except Exception as e:
    logging.error(f"FATAL: An unexpected error occurred during model loading: {e}")
    exit(1)

# --- END OF CRITICAL SECTION ---


# Initialize Flask App
app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Prometheus counter for /predict requests
predict_requests_total = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    ["version"] # Label for the service version
)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict restaurant review sentiment
    ---
    tags:
      - Sentiment Analysis
    parameters:
      - name: input
        in: body
        required: true
        schema:
          type: object
          required:
            - review
          properties:
            review:
              type: string
              example: "The food was amazing and the service was great!"
    responses:
      200:
        description: Prediction result
        examples:
          application/json:
            prediction: "positive"
            version: "1.0"
    """
    try:
        data = request.get_json()
        if not data or "review" not in data:
            return jsonify({"error": "Missing 'review' in request body"}), 400

        review = data["review"].strip()
        if not review:
            return jsonify({"error": "'review' field cannot be empty"}), 400

        logging.info(f"Received review: '{review}'")

        processed_text = preprocess_text(review)
        features = vectorizer.transform([processed_text]).toarray()
        prediction = int(model.predict(features)[0])
        sentiment = "positive" if prediction == 1 else "negative"

        logging.info(f"Prediction successful, sentiment: {sentiment}")

        # Increment Prometheus counter with the service version
        predict_requests_total.labels(version=SERVICE_VERSION).inc()

        return jsonify({"prediction": sentiment, "version": SERVICE_VERSION})

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

@app.route("/metrics")
def metrics():
    """Endpoint for Prometheus to scrape metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/version", methods=["GET"])
def version():
    """Returns the version of the service."""
    return jsonify({"version": SERVICE_VERSION})

@app.route('/health', methods=['GET'])
def health():
    """A simple health check endpoint for Kubernetes probes."""
    # A more advanced check could verify model loading, but for now this is fine.
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    # This block is for local development, not used by 'flask run' or Gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
