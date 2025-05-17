FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY service/ service/
COPY . .


EXPOSE 8000

CMD ["python", "service/app.py"]
