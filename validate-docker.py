#!/usr/bin/env python3
"""
Docker deployment validation script.
Checks if all required files and directories are properly configured.
"""

import os
import sys
import platform
from pathlib import Path

def check_exists(path, description):
    """Check if path exists and print status."""
    if Path(path).exists():
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} (NOT FOUND)")
        return False

def get_data_path():
    """Get data path based on environment variable or OS default."""
    # Try to load from .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("DATA_PATH="):
                    path_value = line.split("=", 1)[1].strip()
                    return Path(path_value)
    
    # Default paths based on OS
    if platform.system() == "Windows":
        return Path("C:/anifind-data")
    else:
        return Path("/data/anifind")

def check_env_file():
    """Check if .env file has required variables."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    required_vars = [
        "MINIO_ENDPOINT",
        "MINIO_ACCESS_KEY", 
        "MINIO_SECRET_KEY",
        "API_HOST",
        "API_PORT"
    ]
    
    with open(env_file) as f:
        content = f.read()
    
    missing_vars = []
    for var in required_vars:
        if f"{var}=" not in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        print("✅ Environment file configured")
        return True

def main():
    """Main validation function."""
    print("🐳 Docker Deployment Validation")
    print("=" * 40)
    print(f"🖥️  Operating System: {platform.system()}")
    
    all_good = True
    
    # Check Docker files
    all_good &= check_exists("docker-compose.yml", "Docker Compose file")
    all_good &= check_exists(".env.example", "Environment template")
    
    # Check environment configuration
    all_good &= check_env_file()
    
    # Get data path based on OS and configuration
    data_path = get_data_path()
    print(f"\n📁 Expected data path: {data_path}")
    
    # Check data directories
    if data_path.exists():
        print("\n� Data directories:")
        all_good &= check_exists(data_path / "indexes", "Indexes directory")
        all_good &= check_exists(data_path / "videos", "Videos directory")
        all_good &= check_exists(data_path / "indexes" / "global_index.faiss", "FAISS index")
        all_good &= check_exists(data_path / "indexes" / "metadata.pkl", "Metadata file")
        
        # Check if directories are writable
        try:
            test_file = data_path / "checkpoints" / "test"
            test_file.parent.mkdir(exist_ok=True)
            test_file.touch()
            test_file.unlink()
            print("✅ Data directories are writable")
        except Exception as e:
            print(f"❌ Data directories not writable: {e}")
            all_good = False
    else:
        print(f"\n⚠️  Data directories not found at {data_path}")
        print("   Please create the directory structure:")
        
        if platform.system() == "Windows":
            print(f"   mkdir {data_path}\\indexes")
            print(f"   mkdir {data_path}\\videos")
            print(f"   mkdir {data_path}\\checkpoints")
            print(f"   mkdir {data_path}\\previews")
        else:
            print(f"   mkdir -p {data_path}/{{indexes,videos,checkpoints,previews}}")
        
        all_good = False
    
    # OS-specific recommendations
    print(f"\n💡 {platform.system()} Setup Recommendations:")
    if platform.system() == "Windows":
        print("   • Use Docker Desktop for Windows")
        print("   • Set DATA_PATH=C:/anifind-data in .env")
        print("   • Make sure Docker has access to C: drive")
        print("   • Remove DOCKER_USER from .env (not needed on Windows)")
    else:
        print("   • Set DATA_PATH=/data/anifind in .env") 
        print("   • Set DOCKER_USER=1000:1000 in .env")
        print("   • Ensure proper permissions: sudo chown -R 1000:1000 /data/anifind")
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 All checks passed! Ready for deployment:")
        print("   docker-compose up -d")
    else:
        print("❌ Some issues found. Please fix them before deployment.")
        
        # Provide helpful setup commands
        print(f"\n🔧 Quick setup for {platform.system()}:")
        if platform.system() == "Windows":
            print("   # Copy and edit configuration")
            print("   copy .env.example .env")
            print("   # Edit .env and set DATA_PATH=C:/anifind-data")
            print("   # Create directories")
            print(f"   mkdir {data_path}\\indexes {data_path}\\videos")
        else:
            print("   # Copy and edit configuration")
            print("   cp .env.example .env")
            print("   # Edit .env and set DATA_PATH=/data/anifind")
            print("   # Create directories")
            print(f"   sudo mkdir -p {data_path}/{{indexes,videos,checkpoints,previews}}")
            print(f"   sudo chown -R 1000:1000 {data_path}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
