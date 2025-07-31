from minio import Minio
from minio.error import S3Error
from pathlib import Path
from datetime import timedelta
import time
from config import config

MINIO_CLIENT = Minio(
    config.MINIO_ENDPOINT,
    access_key=config.MINIO_ACCESS_KEY,
    secret_key=config.MINIO_SECRET_KEY,
    secure=config.MINIO_SECURE
)

BUCKET_NAME = config.MINIO_BUCKET_NAME

def ensure_bucket():
    """Create bucket if it doesn't exist."""
    if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
        MINIO_CLIENT.make_bucket(BUCKET_NAME)
        print(f"[✅] Bucket '{BUCKET_NAME}' created")

def minio_object_exists(object_name: str) -> bool:
    try:
        MINIO_CLIENT.stat_object(BUCKET_NAME, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise

def upload_preview(local_path: Path, anime: str, filename: str) -> str:
    """
    Upload preview video to MinIO and return presigned URL (valid for 24h).
    
    Args:
        local_path: Path to the local video file
        anime: Anime name (used for folder organization)
        filename: Target filename
        
    Returns:
        Presigned URL valid for 24 hours
    """
    ensure_bucket()
    object_name = f"{anime}/{filename}"
    print(f"[🚀] Checking MinIO for: {object_name}")
    
    if not minio_object_exists(object_name):
        print(f"[⬆️] Uploading: {local_path} → {object_name}")
        
        # Measure upload time
        upload_start_time = time.time()
        MINIO_CLIENT.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=str(local_path),
            content_type="video/mp4"
        )
        upload_end_time = time.time()
        upload_duration = upload_end_time - upload_start_time
        
        # Get file size for upload rate calculation
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        upload_rate = file_size_mb / upload_duration if upload_duration > 0 else 0
        
        print(f"[✅] Upload completed: {object_name}")
        print(f"[⏱️] Upload time: {upload_duration:.2f}s | Size: {file_size_mb:.1f}MB | Rate: {upload_rate:.1f}MB/s")
    else:
        print(f"[📦] Already exists on MinIO: {object_name}")
    
    # Generate presigned URL (valid for configured hours)
    try:
        url_start_time = time.time()
        presigned_url = MINIO_CLIENT.presigned_get_object(
            BUCKET_NAME, 
            object_name, 
            expires=timedelta(hours=config.PREVIEW_URL_EXPIRES_HOURS)
        )
        url_end_time = time.time()
        url_duration = url_end_time - url_start_time
        
        print(f"[🔗] Generated presigned URL (valid for {config.PREVIEW_URL_EXPIRES_HOURS}h)")
        print(f"[⏱️] URL generation time: {url_duration:.3f}s")
        return presigned_url
    except Exception as e:
        print(f"[❌] Error generating presigned URL: {e}")
        raise


def get_presigned_url(object_name: str, expires_hours: int = 24) -> str:
    """
    Generate presigned URL for an object in MinIO.
    
    Args:
        object_name: Object name in the bucket
        expires_hours: URL expiration time in hours
        
    Returns:
        Presigned URL valid for the specified time
    """
    try:
        return MINIO_CLIENT.presigned_get_object(
            BUCKET_NAME, 
            object_name, 
            expires=timedelta(hours=expires_hours)
        )
    except Exception as e:
        print(f"[❌] Error generating presigned URL: {e}")
        raise

