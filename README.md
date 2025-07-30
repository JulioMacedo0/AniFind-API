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

### 2. Run the API

```bash
# Run with default settings
uv run run_api.py

# Run in development mode (with auto-reload)
uv run run_api.py --reload

# Run on specific host/port
uv run run_api.py --host 0.0.0.0 --port 8080

# Run with custom workers
uv run run_api.py --workers 4
```

### 3. Access documentation

After starting the API, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

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

You can configure the API using environment variables:

```bash
# API Configuration
export ANIFIND_HOST=0.0.0.0
export ANIFIND_PORT=8000
export ANIFIND_WORKERS=1

# Data Paths
export FAISS_INDEX_PATH=indexes/global_index.faiss
export METADATA_PATH=indexes/metadata.pkl

# MinIO Configuration (for preview uploads)
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=your_access_key
export MINIO_SECRET_KEY=your_secret_key
```

## Performance

- **Startup time**: ~2-5 seconds (loading FAISS index and metadata)
- **Search time**: ~20-50ms per query
- **Memory usage**: ~1-3GB (depending on index size)
- **Concurrent requests**: Supports multiple simultaneous searches

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful search
- `400`: Invalid image file or format
- `404`: File not found
- `500`: Internal server error
- `503`: Service unavailable (data not loaded)

## License

This project is licensed under the MIT License.
