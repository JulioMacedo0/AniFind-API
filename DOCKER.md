# Docker Deployment Guide

Simple guide for deploying AniFind API with Docker.

## Prerequisites

1. **Generated data on powerful machine:**

   ```bash
   uv run createPhash.py          # Generate FAISS indexes
   uv run migrate_metadata.py     # Ensure relative paths
   ```

2. **Docker installed on production server**

## Setup

### 1. Prepare Production Server

#### Windows

```batch
# Create data directories
mkdir C:\anifind-data\indexes
mkdir C:\anifind-data\videos
mkdir C:\anifind-data\checkpoints
mkdir C:\anifind-data\previews
```

#### Linux/Mac

```bash
# Create data directories
sudo mkdir -p /data/anifind/{indexes,videos,checkpoints,previews}
sudo chown -R 1000:1000 /data/anifind/
```

### 2. Transfer Data

**From powerful machine:**

```bash
tar -czf anifind-data.tar.gz indexes/ test/
```

**On production server:**

#### Windows

```batch
# Extract to C:\anifind-data\
# Move contents of test\ to videos\
move C:\anifind-data\test\* C:\anifind-data\videos\
rmdir C:\anifind-data\test
```

#### Linux/Mac

```bash
cd /data/anifind/
sudo tar -xzf /tmp/anifind-data.tar.gz
sudo mv test/* videos/
sudo rm -rf test/
sudo chown -R 1000:1000 /data/anifind/
```

### 3. Configure & Deploy

```bash
# Clone repository
git clone your-repo-url
cd anime-episode-finder

# Configure environment
cp .env.example .env
# Edit .env with your settings:
#   Windows: DATA_PATH=C:/anifind-data (and comment out DOCKER_USER)
#   Linux/Mac: DATA_PATH=/data/anifind (and keep DOCKER_USER=1000:1000)

# Deploy
docker-compose up -d

# Check status
docker-compose logs -f
curl http://localhost:8000/health
```

## Volume Mapping

The docker-compose.yml maps these directories based on DATA_PATH:

- `./:/app:ro` → Application code (read-only)
- `${DATA_PATH}/indexes:/app/indexes` → FAISS indexes and metadata
- `${DATA_PATH}/videos:/app/videos` → Original video files
- `${DATA_PATH}/checkpoints:/app/checkpoints` → Processing cache
- `${DATA_PATH}/previews:/app/previews` → Preview cache

**Examples:**

- Windows: `C:/anifind-data/indexes:/app/indexes`
- Linux: `/data/anifind/indexes:/app/indexes`

## Environment Variables

All configuration is done via `.env` file:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2

# MinIO
MINIO_ENDPOINT=your-minio-server.com
MINIO_ACCESS_KEY=your-key
MINIO_SECRET_KEY=your-secret
MINIO_SECURE=true

# Paths (container paths)
FAISS_INDEX_PATH=/app/indexes/global_index.faiss
METADATA_PATH=/app/indexes/metadata.pkl
VIDEO_BASE_DIR=/app/videos
```

## Monitoring

```bash
# Check logs
docker-compose logs anifind-api

# Check health
curl http://localhost:8000/health

# Check disk usage
df -h /data/anifind/
du -sh /data/anifind/*
```

That's it! Your AniFind API is now running in production.
