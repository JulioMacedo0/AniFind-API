FROM python:3.13-slim

# Install system dependencies as root
RUN apt-get update && \
    apt-get install -y ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv as root
RUN pip install uv

# Create app directory and set permissions
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --no-dev

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/checkpoints /app/indexes /app/previews /app/videos && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "run_api.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
