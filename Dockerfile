FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY webscrapper_api.py .

EXPOSE 8000

CMD ["uvicorn", "webscrapper_api:app", "--host", "0.0.0.0", "--port", "8000"]