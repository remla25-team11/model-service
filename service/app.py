import os
import requests
import joblib
import logging
from flask import Flask, request, jsonify
from flasgger import Swagger
from flask_cors import CORS 
from lib_ml.preprocessor import preprocess_text  # Ensure this is correct
from lib_ml import __version__ as lib_ml_version


app = Flask(__name__)
swagger = Swagger(app)
CORS(app) 

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

      result = int(model.predict(features)[0])  # convert numpy int64 to native int
      sentiment = "positive" if result == 1 else "negative"

      logging.info("Prediction succesful, prediction: " + sentiment)
      return jsonify({"prediction": sentiment})
    except Exception as e:
        logging.exception("Failed prediction")
        return jsonify({"error": str(e)}), 500
    

@app.route("/version", methods=["GET"])
def version():
    """
    Returns the version of the model-service and lib_ml package.
    ---
    tags:
      - Metadata
    responses:
      200:
        description: Version information
        schema:
          type: object
          properties:
            model_service_version:
              type: string
              example: "1.0.0"
            lib_ml_version:
              type: string
              example: "1.0.0"
    """
    logging.info("Returned model service version: " + MODEL_SERVICE_VERSION)
    logging.info("Returned mlib-ml version: " + lib_ml_version)
    return jsonify({
        "model_service_version": MODEL_SERVICE_VERSION or "unknown, probs bug, please check",
        "lib_ml_version": lib_ml_version
    })



@app.route("/version", methods=["GET"])
def version():
    #Get the model version from the model-training repository's latest GitHub release tag.

    github_api_url = "https://api.github.com/repos/remla25-team11/model-training/releases/latest"
    try:
        response = requests.get(github_api_url)
        response.raise_for_status() # Raise an exception for HTTP errors
        release_info = response.json()
        model_version = release_info.get("tag_name", "unknown")
        return jsonify({"version": model_version}), 200
    except requests.exceptions.RequestException as e:
        print(f"Error fetching model version from GitHub API: {e}")
        return jsonify({"error": "Could not fetch model version", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
