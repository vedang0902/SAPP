# Atmospheric Monitoring Pipeline - Docker Image
# Python 3.10 slim base
# Dependencies: tensorflow (LSTM), statsmodels (SARIMA) for hybrid forecasting
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install dependencies (tensorflow, statsmodels, scikit-learn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

RUN mkdir -p data output logs models

EXPOSE 8000

# Run main pipeline by default
CMD ["python", "app.py"]
