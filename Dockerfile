FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# SQLite DB + signal files live on a mounted volume at /app/data
VOLUME ["/app/data"]

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "idxquant.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
