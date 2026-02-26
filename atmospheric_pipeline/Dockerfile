# Atmospheric Monitoring Pipeline - Docker Image
# Python 3.10 slim base
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run main pipeline by default
CMD ["python", "main_pipeline.py"]
