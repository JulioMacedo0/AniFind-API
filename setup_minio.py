#!/usr/bin/env python3
"""
Simple MinIO setup script for AniFind.
This script just creates the bucket needed for video previews.
"""

from minio import Minio
from minio.error import S3Error
import sys

def setup_minio():
    """Setup MinIO bucket for previews."""
    
    # MinIO client configuration
    client = Minio(
        "localhost:9000",
        access_key="admin",
        secret_key="admin123",
        secure=False
    )
    
    bucket_name = "previews"
    
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
    
    # Check if MinIO is running
    try:
        client = Minio(
            "localhost:9000",
            access_key="admin",
            secret_key="admin123",
            secure=False
        )
        
        # Test connection
        client.list_buckets()
        print("✅ MinIO connection successful")
        
    except Exception as e:
        print(f"❌ Cannot connect to MinIO: {e}")
        print("\n💡 Make sure MinIO is running:")
        print("   docker run -p 9000:9000 -p 9001:9001 \\")
        print("     -e MINIO_ROOT_USER=admin \\")
        print("     -e MINIO_ROOT_PASSWORD=admin123 \\")
        print("     minio/minio server /data --console-address ':9001'")
        return False
    
    # Setup MinIO
    return setup_minio()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
