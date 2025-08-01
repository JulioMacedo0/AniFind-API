# AniFind API

FastAPI-based API for finding anime episodes through image search using perceptual hashing.

## Prerequisites

- Python 3.13+
- UV package manager
- FAISS indexes and metadata in `indexes/` directory
- MinIO configuration for preview uploads

## Setup

### 1. Install dependencies

```bash
# Install dependencies using UV
uv sync

# Or install in development mode
uv sync --dev
```

### 2. Configure environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configuration
nano .env
```

**Environment Variables:**

- `MINIO_ENDPOINT` - MinIO server endpoint (default: localhost:9000)
- `MINIO_ACCESS_KEY` - MinIO access key (default: admin)
- `MINIO_SECRET_KEY` - MinIO secret key (default: admin123)
- `MINIO_SECURE` - Use HTTPS for MinIO (default: false)
- `MINIO_BUCKET_NAME` - Bucket name for previews (default: previews)
- `API_HOST` - API host (default: 127.0.0.1)
- `API_PORT` - API port (default: 8000)
- `FAISS_INDEX_PATH` - Path to FAISS index file
- `METADATA_PATH` - Path to metadata pickle file
- `SEARCH_TOP_K` - Number of top results to return (default: 3)
- `PREVIEW_URL_EXPIRES_HOURS` - Presigned URL expiration in hours (default: 24)

### 3. Setup MinIO (for video previews)

```bash
# Start MinIO server (using Docker)
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=admin \
  -e MINIO_ROOT_PASSWORD=admin123 \
  minio/minio server /data --console-address ':9001'

# Configure MinIO bucket and policies
uv run setup_minio.py

# Access MinIO Console (optional)
# http://localhost:9001 (admin/admin123)
```

### 4. Run the API

```bash
# Run with default settings (from .env)
uv run run_api.py

# Run in development mode (with auto-reload)
uv run run_api.py --reload

# Run on specific host/port (overrides .env)
uv run run_api.py --host 0.0.0.0 --port 8080

# Run with custom workers
uv run run_api.py --workers 4
```

### 5. Access documentation

After starting the API, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Quick Start

### Local Development

1. Install dependencies: `uv sync`
2. Configure environment: `cp .env.example .env`
3. Start MinIO: `docker run -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=admin -e MINIO_ROOT_PASSWORD=admin123 minio/minio server /data --console-address ':9001'`
4. Setup MinIO: `uv run setup_minio.py`
5. Start API: `uv run run_api.py --reload`
6. Test: Open http://localhost:8000/docs

### Docker Production

1. Configure environment: `cp .env.example .env` and edit your settings
2. Prepare data structure (see [Docker Deployment](#docker-deployment) section)
3. Run: `docker-compose up -d`
4. Check: `curl http://localhost:8000/health`

## Docker Deployment

### Data Structure

Your production server needs this directory structure:

```
/data/anifind/
├── indexes/
│   ├── global_index.faiss    # Generated on powerful machine
│   └── metadata.pkl          # Generated on powerful machine
├── videos/                   # Original video files
│   ├── anime1.mkv
│   └── anime2.mkv
├── checkpoints/              # Auto-created cache
└── previews/                 # Auto-created cache
```

### Setup Steps

1. **Copy data from generation machine:**

   ```bash
   # On powerful machine
   tar -czf anifind-data.tar.gz indexes/ test/
   scp anifind-data.tar.gz user@server:/tmp/

   # On production server
   sudo mkdir -p /data/anifind/{indexes,videos,checkpoints,previews}
   sudo tar -xzf /tmp/anifind-data.tar.gz -C /data/anifind/
   sudo mv /data/anifind/test/* /data/anifind/videos/
   sudo chown -R 1000:1000 /data/anifind/
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your MinIO credentials and settings
   ```

3. **Deploy:**
   ```bash
   docker-compose up -d
   docker-compose logs -f
   ```

## Available Endpoints

### `POST /api/v1/search`

Search for anime episode using an image.

**Parameters:**

- `image`: Image file (JPG, PNG, BMP, TIFF, WEBP)

**Response:**

```json
{
  "query": "path/to/image.png",
  "top_result": {
    "rank": 1,
    "anime": "One Piece",
    "season": 1,
    "episode": 15,
    "timecode": "12:34",
    "second": 754.5,
    "similarity": 95.2,
    "anime_id": "one_piece",
    "source_file": "OnePiece_S01E15.mkv",
    "preview_source_path": "/path/to/video.mkv",
    "preview_video": "https://preview-url.com/video.mp4"
  },
  "all_results": [
    {
      "rank": 1,
      "anime": "One Piece",
      "season": 1,
      "episode": 15,
      "timecode": "12:34",
      "second": 754.5,
      "similarity": 95.2,
      "anime_id": "one_piece",
      "source_file": "OnePiece_S01E15.mkv",
      "preview_source_path": "/path/to/video.mkv"
    },
    {
      "rank": 2,
      "anime": "One Piece",
      "season": 1,
      "episode": 14,
      "timecode": "08:15",
      "second": 495.0,
      "similarity": 87.3,
      "anime_id": "one_piece",
      "source_file": "OnePiece_S01E14.mkv",
      "preview_source_path": "/path/to/video.mkv"
    }
  ],
  "preview_url": "https://preview-url.com/video.mp4"
}
```

### `GET /api/v1/health`

Check if the search service is operational.

**Response:**

```json
{
  "status": "healthy",
  "message": "Search service operational",
  "stats": {
    "index_size": 150000,
    "metadata_entries": 150000
  }
}
```

### `GET /api/v1/stats`

Get detailed service statistics and information.

**Response:**

```json
{
  "service_name": "AniFind API",
  "version": "1.0.0",
  "status": "operational",
  "uptime": "2h 30m 15s",
  "data_status": {
    "initialized": true,
    "index_loaded": true,
    "metadata_loaded": true,
    "index_size": 150000,
    "metadata_entries": 150000
  },
  "performance": {
    "average_search_time": 0.045,
    "total_searches": 1250,
    "cache_hit_rate": 98.5
  }
}
```

### `GET /health`

Basic health check endpoint.

## Usage Examples

### Using curl

```bash
# Search for anime episode
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@frame.png"

# Check service health
curl http://localhost:8000/api/v1/health

# Get service statistics
curl http://localhost:8000/api/v1/stats
```

### Using Python

```python
import requests

# Search for anime episode
url = "http://localhost:8000/api/v1/search"
files = {"image": open("frame.png", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Anime: {result['top_result']['anime']}")
print(f"Episode: S{result['top_result']['season']:02d}E{result['top_result']['episode']:02d}")
print(f"Similarity: {result['top_result']['similarity']:.1f}%")
print(f"Timecode: {result['top_result']['timecode']}")

# Check service status
health_response = requests.get("http://localhost:8000/api/v1/health")
print(f"Service status: {health_response.json()['status']}")
```

### Using JavaScript/Fetch

```javascript
// Search for anime episode
const formData = new FormData();
formData.append("image", fileInput.files[0]);

fetch("http://localhost:8000/api/v1/search", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    console.log("Anime:", data.top_result.anime);
    console.log(
      "Episode:",
      `S${data.top_result.season
        .toString()
        .padStart(2, "0")}E${data.top_result.episode
        .toString()
        .padStart(2, "0")}`
    );
    console.log("Similarity:", `${data.top_result.similarity.toFixed(1)}%`);
  });
```

## Development

### Project Structure

```
anime-episode-finder/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Main FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── image_search_models.py  # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── image_search.py         # API routes
│   └── services/
│       ├── __init__.py
│       └── image_search_service.py # Business logic
├── indexes/
│   ├── global_index.faiss          # FAISS search index
│   └── metadata.pkl                # Episode metadata
├── searchPhash.py                  # Core search functionality
├── run_api.py                      # API runner script
├── pyproject.toml                  # Project dependencies
└── uv.lock                         # Locked dependencies
```

### Running Tests

```bash
# Run the test script
uv run test_anifind.py

# Run example usage
uv run example_usage.py
```

### Environment Variables

All configuration is managed through environment variables in the `.env` file:

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin123
MINIO_SECURE=false
MINIO_BUCKET_NAME=previews
PREVIEW_URL_EXPIRES_HOURS=24

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=1

# Data Paths
FAISS_INDEX_PATH=indexes/global_index.faiss
METADATA_PATH=indexes/metadata.pkl
SEARCH_TOP_K=3
```

## Project Structure

```
anime-episode-finder/
├── app/
│   ├── __init__.py
│   ├── main.py                     # Main FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── image_search_models.py  # Pydantic models
│   ├── routers/
│   │   ├── __init__.py
│   │   └── image_search.py         # API routes
│   └── services/
│       ├── __init__.py
│       └── image_search_service.py # Business logic
├── indexes/
│   ├── global_index.faiss          # FAISS search index
│   └── metadata.pkl                # Episode metadata
├── config.py                       # Configuration management
├── searchPhash.py                  # Core search functionality
├── minio_client.py                 # MinIO integration
├── setup_minio.py                  # MinIO setup script
├── run_api.py                      # API runner script
├── .env.example                    # Environment variables template
├── pyproject.toml                  # Project dependencies
└── uv.lock                         # Locked dependencies
```

## Performance

- **Startup time**: ~2-5 seconds (loading FAISS index and metadata)
- **Search time**: ~20-50ms per query
- **Memory usage**: ~1-3GB (depending on index size)
- **Concurrent requests**: Supports multiple simultaneous searches
- **Data persistence**: Index and metadata loaded once at startup

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful search
- `400`: Invalid image file or format
- `404`: File not found
- `500`: Internal server error
- `503`: Service unavailable (data not loaded)

## License

This project is licensed under the MIT License.
