FROM python:3.12-slim

# Default version
ARG SERVICE_VERSION=unknown

ENV SERVICE_VERSION=$SERVICE_VERSION
ENV FLASK_APP=service/app.py

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY service/ service/
COPY . .

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
