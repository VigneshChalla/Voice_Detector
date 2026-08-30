FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY voice_detection_app/ ./voice_detection_app/
COPY sdk/ ./sdk/
COPY train_local.py .

# Create data directories
RUN mkdir -p data/enrollments data/feature_logs data/audit_logs data/compliance_logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "voice_detection_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
