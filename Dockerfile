FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch first, then fer (prevents fer from pulling GPU torch + CUDA libs)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fer>=22.5.1

# Copy application code
COPY . .

# Run the application (use PORT env var provided by Railway)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
