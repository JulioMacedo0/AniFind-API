#!/usr/bin/env python3
"""
Simple MinIO setup script for AniFind.
This script just creates the bucket needed for video previews.
"""

from minio import Minio
from minio.error import S3Error
import sys
from config import config

def setup_minio():
    """Setup MinIO bucket for previews."""
    
    # MinIO client configuration from environment
    client = Minio(
        config.MINIO_ENDPOINT,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=config.MINIO_SECURE
    )
    
    bucket_name = config.MINIO_BUCKET_NAME
    
    try:
        # Create bucket if it doesn't exist
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✅ Bucket '{bucket_name}' created successfully")
        else:
            print(f"ℹ️  Bucket '{bucket_name}' already exists")
        
        # Test the setup
        print("\n🧪 Testing MinIO setup...")
        
        # List buckets
        buckets = client.list_buckets()
        print(f"📦 Available buckets: {[bucket.name for bucket in buckets]}")
        
        print("\n🎉 MinIO setup completed successfully!")
        print("📝 Videos will be accessible via presigned URLs (24h expiry)")
        
        return True
        
    except S3Error as e:
        print(f"❌ MinIO S3 Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def main():
    print("🚀 AniFind MinIO Setup")
    print("=" * 40)
    
    # Print current configuration
    config.print_config()
    print()
    
    # Check if MinIO is running
    try:
        client = Minio(
            config.MINIO_ENDPOINT,
            access_key=config.MINIO_ACCESS_KEY,
            secret_key=config.MINIO_SECRET_KEY,
            secure=config.MINIO_SECURE
        )
        
        # Test connection
        client.list_buckets()
        print("✅ MinIO connection successful")
        
    except Exception as e:
        print(f"❌ Cannot connect to MinIO: {e}")
        print("\n💡 Make sure MinIO is running:")
        print(f"   docker run -p 9000:9000 -p 9001:9001 \\")
        print(f"     -e MINIO_ROOT_USER={config.MINIO_ACCESS_KEY} \\")
        print(f"     -e MINIO_ROOT_PASSWORD={config.MINIO_SECRET_KEY} \\")
        print(f"     minio/minio server /data --console-address ':9001'")
        return False
    
    # Setup MinIO
    return setup_minio()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
