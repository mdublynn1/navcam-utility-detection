# Base: Official Ultralytics image (CPU build)
FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

# System deps: ffmpeg for video frame extraction
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Copy API source
COPY main.py .

# Expose API port
EXPOSE 8001

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
