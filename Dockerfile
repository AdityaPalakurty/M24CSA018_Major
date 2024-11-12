# Dockerfile
"""
# Use official Python base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app files
COPY . .

# Expose port for MLflow tracking server
EXPOSE 5000

# Run main app
CMD ["python", "code.py"]
"""

# docker-compose.yml
"""
version: '3.7'

services:
  mlapp:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    volumes:
      - ./mlruns:/mlruns
  mlflow:
    image: mlflow/mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://0.0.0.0:5000
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns