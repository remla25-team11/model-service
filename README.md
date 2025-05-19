# Model Service

This is a Flask-based microservice for predicting sentiment in restaurant reviews. It exposes an HTTP API endpoint (`/predict`) and uses a pre-trained model + vectorizer released in the [`model-training`](https://github.com/remla25-team11/model-training) repository.

---

## Features

- Uses a reusable text preprocessor from [`lib-ml`](https://github.com/remla25-team11/lib-ml)
- Downloads model and vectorizer dynamically at startup
- Swagger UI for testing the `/predict` endpoint
- Fully containerized using Docker
- Automated GitHub Actions workflow pushes versioned Docker images to GHCR

---


## How to Run

### Enviroment variable
- 'MODEL_URL': URL to access the trained mode file
- 'VECTORIZER_URL': URL to download the trained vectorizer
- 'PORT': (optional, default: 8000) The port of the Flask server
- 'SERVICE_VERSION': injected, passed form docker build

### Run with Docker (recommended)

```bash
docker run -p 8000:8000 \
  -e MODEL_URL=https://github.com/remla25-team11/model-training/releases/download/v0.0.1/c2_Classifier_Sentiment_Model \
  -e VECTORIZER_URL=https://github.com/remla25-team11/model-training/releases/download/v0.0.1/c1_BoW_Sentiment_Model.pkl \
  ghcr.io/remla25-team11/model-service:latest
```
### API endpoints

- POST /predict
- GET /version
