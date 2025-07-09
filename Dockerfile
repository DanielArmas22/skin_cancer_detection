# Base image - Python 3.11 para mejor compatibilidad con TF 2.18
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow, OpenCV and emoji fonts
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-noto-color-emoji \
    fonts-noto-cjk \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app/ ./app/

# Expose port
EXPOSE 8501

# Set environment variables for TensorFlow and emoji support
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONPATH=/app
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Update font cache
RUN fc-cache -fv

# Command to run the application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
