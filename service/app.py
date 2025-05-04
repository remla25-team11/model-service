import os
import requests
import joblib
from flask import Flask, request, jsonify
from flasgger import Swagger
from lib_ml.preprocessor import preprocess_text  # Ensure this is correct

app = Flask(__name__)
swagger = Swagger(app)

# Define model paths for local
MODEL_PATH = "service/model.joblib" #change to your path
VECTORIZER_PATH = "service/vectorizer.pkl" #change to your path

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
