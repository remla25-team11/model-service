import os
import json
import requests
import joblib
import logging
from flask import Flask, request, jsonify, Response
from flasgger import Swagger
from flask_cors import CORS
from lib_ml.preprocessor import preprocess_text
from lib_ml import __version__ as lib_ml_version

# Prometheus client
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)
swagger = Swagger(app)
CORS(app)

# Prometheus counter
predict_requests_total = Counter("predict_requests_total", "Total number of /predict requests", ["version"])

# Define model paths for local
MODEL_PATH = "service/service/model.joblib"
VECTORIZER_PATH = "service/service/vectorizer.pkl"

# URLs from environment
MODEL_URL = os.getenv("MODEL_URL")
VECTORIZER_URL = os.getenv("VECTORIZER_URL")
MODEL_SERVICE_VERSION = os.getenv("SERVICE_VERSION")

# Basic logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# Download helper
def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        logging.info(f"Downloading {url}...")
        r = requests.get(url)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)

# Download model/vectorizer if missing
if MODEL_URL:
    download_file(MODEL_URL, MODEL_PATH)
if VECTORIZER_URL:
    download_file(VECTORIZER_URL, VECTORIZER_PATH)

# Load once
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

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
    """
    try:
        data = request.get_json()
        review = data.get("review", "").strip()
        logging.info("Received review: " + review)

        if review == "" or review is None:
            return jsonify({"error": "Missing 'review' field"}), 400

        processed = preprocess_text(review)
        features = vectorizer.transform([processed]).toarray()

        result = int(model.predict(features)[0])
        sentiment = "positive" if result == 1 else "negative"

        logging.info("Prediction successful, prediction: " + sentiment)

        # Increment Prometheus counter
        predict_requests_total.labels(version=os.getenv("SERVICE_VERSION", "unknown")).inc()

        return jsonify({"prediction": sentiment})
    except Exception as e:
        logging.exception("Failed prediction")
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/version", methods=["GET"])
def version():
    github_api_url = "https://api.github.com/repos/remla25-team11/model-training/tags"
    try:
        response = requests.get(github_api_url)
        response.raise_for_status()
        raw_json = response.json()

        if isinstance(raw_json, str):
            tags = json.loads(raw_json)
        else:
            tags = raw_json

        latest_version = tags[0]["name"] if tags and "name" in tags[0] else "unknown"
        return jsonify({"version": latest_version})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching model version from GitHub API: {e}")
        return jsonify({"error": "Could not fetch model version", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
