# Dockerfile

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


