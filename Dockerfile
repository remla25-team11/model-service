# model-service/Dockerfile

FROM python:3.12-slim-bookworm

WORKDIR /app

# Arguments that can be passed in at build time
ARG MODEL_URL
ARG VECTORIZER_URL

# Environment variables
ARG SERVICE_VERSION=unknown
ENV SERVICE_VERSION=$SERVICE_VERSION
ENV FLASK_APP=service/app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY model-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === THIS IS THE FIX ===
# Pre-download the NLTK data during the build process.
# This makes the image self-contained and removes the runtime internet dependency.
RUN python -m nltk.downloader stopwords

# Create the directory for the models
RUN mkdir -p /app/models

# Download the models from the GitHub release URLs
RUN curl -L -o /app/models/c2_Classifier_Sentiment_Model.pkl "${MODEL_URL}"
RUN curl -L -o /app/models/c1_BoW_Sentiment_Model.pkl "${VECTORIZER_URL}"

# Copy your application code
COPY model-service/service/ ./service/

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
