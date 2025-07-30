from minio import Minio
from minio.error import S3Error
from pathlib import Path
from datetime import timedelta

MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin123",
    secure=False
)

BUCKET_NAME = "previews"

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
        MINIO_CLIENT.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=str(local_path),
            content_type="video/mp4"
        )
        print(f"[✅] Upload completed: {object_name}")
    else:
        print(f"[📦] Already exists on MinIO: {object_name}")
    
    # Generate presigned URL (valid for 24 hours)
    try:
        presigned_url = MINIO_CLIENT.presigned_get_object(
            BUCKET_NAME, 
            object_name, 
            expires=timedelta(hours=24)
        )
        print(f"[🔗] Generated presigned URL (valid for 24h)")
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

