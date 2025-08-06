#!/usr/bin/env python3
"""
Script to run the AniFind FastAPI application.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8080
    python run_api.py --reload  # For development
"""

import uvicorn
import argparse
import logging
from pathlib import Path
import sys
from config import config

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


# Configure logging to filter out health check requests
class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("/health") == -1


def setup_logging():
    """Setup logging configuration to filter health checks."""
    # Apply filter to uvicorn access logger
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.addFilter(HealthCheckFilter())


def main():
    parser = argparse.ArgumentParser(description="Run AniFind API")
    parser.add_argument("--host", default=config.API_HOST, help=f"Host to bind (default: {config.API_HOST})")
    parser.add_argument("--port", type=int, default=config.API_PORT, help=f"Port to bind (default: {config.API_PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=config.API_WORKERS, help=f"Number of workers (default: {config.API_WORKERS})")
    
    args = parser.parse_args()
    
    # Setup logging configuration
    setup_logging()
    
    print(f"🚀 Starting AniFind API...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"👥 Workers: {args.workers}")
    print(f"📖 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🔍 Health check: http://{args.host}:{args.port}/health")
    print(f"🔇 Health check logs: Filtered (clean logs)")
    print("-" * 50)
    
    # Display environment configuration
    print("⚙️  Environment Configuration:")
    print(f"   🎯 Search Results: {config.SEARCH_TOP_K} result(s)")
    print(f"   📊 Minimum Similarity: {config.MINIMUM_SIMILARITY}%")
    print(f"   🗂️  FAISS Index: {config.FAISS_INDEX_PATH}")
    print(f"   📋 Metadata Path: {config.METADATA_PATH}")
    print(f"   🎬 Video Base Dir: {config.VIDEO_BASE_DIR}")
    print(f"   ☁️  MinIO Endpoint: {config.MINIO_ENDPOINT}")
    print(f"   🪣 MinIO Bucket: {config.MINIO_BUCKET_NAME}")
    print(f"   🔒 MinIO Secure: {config.MINIO_SECURE}")
    print(f"   ⏱️  Preview Expires: {config.PREVIEW_URL_EXPIRES_HOURS}h")
    print(f"   🎥 Video Processing: {config.VIDEO_PROCESSING_WIDTH}px, {config.VIDEO_PROCESSING_FPS}fps")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        access_log=True
    )


if __name__ == "__main__":
    main()
