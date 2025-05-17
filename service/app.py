import os
import requests
import joblib
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

# Download helper
def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {url}...")
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
    data = request.get_json()
    review = data.get("review", "")
    if review == "" or review is None:
      return jsonify({"error": "Missing 'review' field"}), 400

    processed = preprocess_text(review)
    features = vectorizer.transform([processed]).toarray()
    result = int(model.predict(features)[0])  # convert numpy int64 to native int
    sentiment = "positive" if result == 1 else "negative"
    return jsonify({"prediction": sentiment})

@app.route("/version", methods=["GET"])
def version():
    """
    Returns the version of the lib_ml package.
    ---
    tags:
      - Metadata
    responses:
      200:
        description: Library version
        schema:
          type: object
          properties:
            version:
              type: string
              example: "1.2.3"
    """
    return jsonify({"version": lib_ml_version})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
