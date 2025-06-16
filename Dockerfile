# model-service/Dockerfile

FROM python:3.12-slim-bookworm

WORKDIR /app

# Arguments that can be passed in at build time
# These will be the URLs to your release assets
ARG MODEL_URL
ARG VECTORIZER_URL

# Environment variables
ARG SERVICE_VERSION=unknown
ENV SERVICE_VERSION=$SERVICE_VERSION
ENV FLASK_APP=service/app.py
ENV FLASK_ENV=production

# Install system dependencies needed for pip git installs and for downloading
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY model-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the directory for the models
RUN mkdir -p /app/models

# Download the models from the GitHub release URLs passed in as build arguments
# Use curl to download the files and place them in the /app/models directory
# The -L flag follows redirects, which GitHub uses for release assets.
RUN curl -L -o /app/models/c2_Classifier_Sentiment_Model.pkl "${MODEL_URL}"
RUN curl -L -o /app/models/c1_BoW_Sentiment_Model.pkl "${VECTORIZER_URL}"

# Copy your application code
COPY model-service/service/ ./service/

EXPOSE 8000
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
