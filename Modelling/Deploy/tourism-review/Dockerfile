FROM python:3.9-slim

ENV PYTHONUNBUFFERED TRUE

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY keys/direct-hope-387806-8d2b15781824.json /app/keys/direct-hope-387806-8d2b15781824.json

COPY review.py .

EXPOSE 8080

ENV CREDENTIALS_PATH=/app/keys/direct-hope-387806-8d2b15781824.json

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 --env CREDENTIALS_PATH=$CREDENTIALS_PATH review:app
