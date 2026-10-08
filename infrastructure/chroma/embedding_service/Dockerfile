FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] sentence-transformers

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002", "--workers", "2"]
