# Atmospheric Monitoring Pipeline - Docker Image
# Python 3.10 slim base
# Dependencies: tensorflow (LSTM), statsmodels (SARIMA) for hybrid forecasting
FROM python:3.10-slim

WORKDIR /app

# Install dependencies (tensorflow, statsmodels, scikit-learn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify critical packages
RUN python -c "import tensorflow; import statsmodels; print('OK: tensorflow, statsmodels')"

# Copy project
COPY . .

# Run main pipeline by default
CMD ["python", "main_pipeline.py"]
