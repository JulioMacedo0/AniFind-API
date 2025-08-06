# Use official uv image with Python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (not the project itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Create non-root user with specific UID/GID and home directory
RUN groupadd -r appuser --gid 1000 && \
    useradd -r -g appuser --uid 1000 --create-home appuser

# Copy application code
COPY . .

# Install project in production mode and set up directories
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked && \
    mkdir -p /app/checkpoints /app/indexes /app/previews /app/videos && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser:appuser /app /home/appuser

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_CACHE_DIR=/home/appuser/.cache/uv
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "run_api.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
