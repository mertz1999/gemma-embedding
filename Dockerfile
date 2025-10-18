# Use Python 3.11 slim bullseye as base image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HOME=/tmp/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (ensure uvicorn/gunicorn/hypercorn are in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create cache directories for model storage
RUN mkdir -p /tmp/transformers_cache /tmp/huggingface /tmp/sentence_transformers

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app

# Switch to the non-root user
USER app

# Expose the port (Keep for documentation)
EXPOSE 8080

# Use an ASGI server binding to 0.0.0.0 and $PORT
# The format "gemaa-embedding:app" means "look for object 'app' in module 'gemaa-embedding.py'"
ENTRYPOINT ["uvicorn", "gemaa-embedding:app", "--host", "0.0.0.0", "--port", "8080"]

##########################################################################
# How to build, run the container, and see logs:
#
# 1. Build the Docker image:
#      docker build -t gemma-embedding .
#
# 2. Run the container:
#      docker run -p 8080:8080 gemma-embedding
#
# 3. View logs (stdout/stderr shown in Docker output):
#      docker logs <container_id>
#    (or if you run interactively, logs will print in your console)
#
# 4. Visit the API (once running):
#      http://localhost:8080/docs
#
##########################################################################