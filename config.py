#!/usr/bin/env python3
"""
Configuration module for AniFind.
Loads environment variables and provides default values.
"""

import os
from pathlib import Path
from typing import Union

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📄 Loaded environment from {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_int_env(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_float_env(key: str, default: float) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_path_env(key: str, default: Union[str, Path]) -> Path:
    """Get path environment variable."""
    return Path(os.getenv(key, str(default)))


class Config:
    """Configuration class for AniFind."""
    
    # === MINIO CONFIGURATION ===
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'admin')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'admin123')
    MINIO_SECURE = get_bool_env('MINIO_SECURE', False)
    MINIO_BUCKET_NAME = os.getenv('MINIO_BUCKET_NAME', 'previews')
    MINIO_MAX_RETRIES = get_int_env('MINIO_MAX_RETRIES', 3)
    MINIO_RETRY_DELAY = get_float_env('MINIO_RETRY_DELAY', 1.0)
    
    # === API CONFIGURATION ===
    API_HOST = os.getenv('API_HOST', '127.0.0.1')
    API_PORT = get_int_env('API_PORT', 8000)
    API_WORKERS = get_int_env('API_WORKERS', 1)
    
    # === DATA PATHS ===
    FAISS_INDEX_PATH = get_path_env('FAISS_INDEX_PATH', 'indexes/global_index.faiss')
    METADATA_PATH = get_path_env('METADATA_PATH', 'indexes/metadata.pkl')
    
    # === VIDEO PATHS (for portable deployments) ===
    VIDEO_BASE_DIR = get_path_env('VIDEO_BASE_DIR', 'test')
    
    # === SEARCH CONFIGURATION ===
    SEARCH_TOP_K = get_int_env('SEARCH_TOP_K', 1)  # Always search for 1 result
    MINIMUM_SIMILARITY = get_float_env('MINIMUM_SIMILARITY', 75.0)  # Minimum similarity threshold
    
    # === VIDEO PROCESSING ===
    VIDEO_PROCESSING_WIDTH = get_int_env('VIDEO_PROCESSING_WIDTH', 512)
    VIDEO_PROCESSING_FPS = get_int_env('VIDEO_PROCESSING_FPS', 6)
    VIDEO_PROCESSING_PIX_FMT = os.getenv('VIDEO_PROCESSING_PIX_FMT', 'rgb24')
    
    # === PREVIEW CONFIGURATION ===
    PREVIEW_URL_EXPIRES_HOURS = get_int_env('PREVIEW_URL_EXPIRES_HOURS', 24)
    
    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("🔧 AniFind Configuration:")
        print(f"   MinIO Endpoint: {cls.MINIO_ENDPOINT}")
        print(f"   MinIO Bucket: {cls.MINIO_BUCKET_NAME}")
        print(f"   MinIO Secure: {cls.MINIO_SECURE}")
        print(f"   API Host: {cls.API_HOST}")
        print(f"   API Port: {cls.API_PORT}")
        print(f"   API Workers: {cls.API_WORKERS}")
        print(f"   FAISS Index: {cls.FAISS_INDEX_PATH}")
        print(f"   Metadata Path: {cls.METADATA_PATH}")
        print(f"   Video Base Dir: {cls.VIDEO_BASE_DIR}")
        print(f"   Search Top K: {cls.SEARCH_TOP_K}")
        print(f"   Minimum Similarity: {cls.MINIMUM_SIMILARITY}%")
        print(f"   Preview URL Expires: {cls.PREVIEW_URL_EXPIRES_HOURS}h")


# Global config instance
config = Config()

if __name__ == "__main__":
    config.print_config()
