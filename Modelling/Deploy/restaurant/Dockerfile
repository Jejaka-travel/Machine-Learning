FROM python:3.9-slim

ENV PYTHONUNBUFFERED TRUE

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model-restaurant/1

COPY model-restaurant/1/ /app/model-restaurant/1/

COPY keys/direct-hope-387806-8d2b15781824.json /app/keys/direct-hope-387806-8d2b15781824.json

COPY app.py .

COPY scalerTarget.joblib /app/scalerTarget.joblib
COPY scalerUser.joblib /app/scalerUser.joblib
COPY scalerItem.joblib /app/scalerItem.joblib

EXPOSE 8080

ENV MODEL_PATH=/app/model-restaurant/1

ENV CREDENTIALS_PATH=/app/keys/direct-hope-387806-8d2b15781824.json

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --env MODEL_PATH=$MODEL_PATH --env CREDENTIALS_PATH=$CREDENTIALS_PATH app:app
