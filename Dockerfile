# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (optimizes caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory and other necessary files
COPY app/ ./app/
COPY model.pkl .
COPY data/bankruptcy_data_tsfresh.csv ./data/

# Expose port 8080 (Cloud Run expects this by default)
EXPOSE 8080

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]