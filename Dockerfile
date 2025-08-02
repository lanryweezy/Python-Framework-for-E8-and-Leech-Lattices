# E8 Leech Lattice Framework - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY e8leech_api/requirements.txt /app/requirements.txt
COPY requirements.txt /app/framework_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -r /app/framework_requirements.txt

# Copy the entire framework
COPY . /app/

# Set Python path to include the framework
ENV PYTHONPATH="/app/src:/app"

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/lattice/health || exit 1

# Run the application
CMD ["python", "e8leech_api/src/main.py"]

