from minio import Minio
from minio.error import S3Error
from pathlib import Path

MINIO_CLIENT = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin123",
    secure=False
)

BUCKET_NAME = "previews"

def ensure_bucket():
    if not MINIO_CLIENT.bucket_exists(BUCKET_NAME):
        MINIO_CLIENT.make_bucket(BUCKET_NAME)

def minio_object_exists(object_name: str) -> bool:
    try:
        MINIO_CLIENT.stat_object(BUCKET_NAME, object_name)
        return True
    except S3Error as e:
        if e.code == "NoSuchKey":
            return False
        raise

def upload_preview(local_path: Path, anime: str, filename: str) -> str:
    ensure_bucket()
    object_name = f"{anime}/{filename}"
    print(f"[🚀] Checking MinIO for: {object_name}")
    if not minio_object_exists(object_name):
        print(f"[⬆️] Uploading: {local_path} → {object_name}")
        MINIO_CLIENT.fput_object(
            BUCKET_NAME, object_name, str(local_path)
        )
    else:
        print(f"[📦] Already exists on MinIO: {object_name}")
    return f"http://localhost:9000/{BUCKET_NAME}/{object_name}"
