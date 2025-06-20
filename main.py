"""
WhatsEpisode API - Main application

This API provides endpoints for anime frame identification using perceptual hash similarity.
It combines the functionality from init.py into a FastAPI application.
"""

import logging
import os
import time
import uvicorn
import numpy as np
import pickle
import faiss
import cv2
from PIL import Image
import imagehash
import psutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whats-episode-api")

# Create the FastAPI application
app = FastAPI(
    title="WhatsEpisode API",
    description="API for finding anime frames using perceptual hash similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables to hold the loaded data
index = None
metadata = None
binary_vectors = None
max_memory = 0

# Path constants
PHASHES_PATH = "database/phashes.npy"
METADATA_PATH = "database/metadata.pkl"

# === Pydantic Models for API Responses ===

class FrameMetadata(BaseModel):
    """Metadata for a frame from an anime episode"""
    anime: str = Field(default="Unknown", description="Anime name")
    season: int = Field(default=1, description="Season number")
    episode: int = Field(default=1, description="Episode number")
    second: int = Field(default=0, description="Second in the episode")
    position: str = Field(default="", description="Position in the second (start, middle, end)")
    filepath: str = Field(default="", description="Path to the source video file")
    time_formatted: str = Field(default="0:00", description="Formatted time (HH:MM:SS or MM:SS)")

class SearchResult(BaseModel):
    """Result of a similarity search"""
    id: int = Field(..., description="ID of the result in the database")
    distance: int = Field(..., description="Hamming distance (number of different bits, 0-64)")
    similarity_percentage: float = Field(..., description="Similarity percentage (100% = identical)")
    metadata: Dict[str, Any] = Field(..., description="Original metadata for the frame")

class SearchTimings(BaseModel):
    """Timing information for the search process"""
    processing_time: float = Field(..., description="Time to process the query image (seconds)")
    search_time: float = Field(..., description="Time to search in the index (seconds)")
    results_time: float = Field(..., description="Time to format the results (seconds)")
    total_time: float = Field(..., description="Total search time (seconds)")

class SearchResponse(BaseModel):
    """Response for the image search endpoint"""
    results: List[SearchResult] = Field(..., description="Search results ordered by similarity")
    timings: SearchTimings = Field(..., description="Timing information for the search process")
    total_results: int = Field(..., description="Number of results returned")
    query_image: str = Field(default="uploaded_image", description="Name of the uploaded query image")

class StatusResponse(BaseModel):
    """API status and memory usage information"""
    status: str = Field(default="ok", description="API status")
    data_loaded: bool = Field(..., description="Whether the index data is loaded")
    index_size: int = Field(..., description="Number of items in the index")
    metadata_count: int = Field(..., description="Number of items in metadata")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")
    max_memory_usage_mb: float = Field(..., description="Peak memory usage in MB")

# === Helper Functions ===

def get_memory_usage():
    """Return the current memory usage of this process in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert bytes to MB

def update_max_memory():
    """Update and return the maximum memory usage"""
    global max_memory
    current = get_memory_usage()
    if current > max_memory:
        max_memory = current
    return current

def compute_phash_vector(frame_bgr):
    """
    Compute perceptual hash vector for an image frame
    
    Args:
        frame_bgr: OpenCV image in BGR format
        
    Returns:
        Binary vector suitable for FAISS IndexBinaryFlat (shape [1, 8])
    """
    # Resize for consistency
    if frame_bgr.shape[0] > 64 or frame_bgr.shape[1] > 64:
        frame_bgr = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    
    # Convert from BGR to RGB for PIL
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    phash_obj = imagehash.phash(img_pil)
    
    # Convert to uint64 format
    hash_uint64 = np.packbits(phash_obj.hash.flatten()).view(np.uint64)[0]
    
    # Convert to binary format for IndexBinaryFlat (array of 8 bytes)
    binary_hash = np.array([hash_uint64], dtype=np.uint64).view(np.uint8)
    
    return binary_hash.reshape(1, 8)  # Format: (1, 8) - one vector of 8 bytes (64 bits)

def search_similar_frame(frame_bgr, k=5):
    """
    Search for similar frames in the FAISS index
    
    Args:
        frame_bgr: OpenCV image in BGR format
        k: Number of results to return
        
    Returns:
        List of results with metadata and similarity scores, plus timing information
    """
    global index, metadata
    
    start_time = time.time()
    logger.info("Starting similarity search")
    
    # Process the query frame
    t_start_proc = time.time()
    query_vector = compute_phash_vector(frame_bgr)
    t_end_proc = time.time()
    proc_time = t_end_proc - t_start_proc
    
    # Search in the FAISS index
    t_start_search = time.time()
    D, I = index.search(query_vector, k)
    t_end_search = time.time()
    search_time = t_end_search - t_start_search
    
    # Format results
    t_start_results = time.time()
    results = []
    for i in range(min(k, len(I[0]))):
        frame_id = int(I[0][i])
        # Hamming distance is the number of different bits (0-64)
        hamming_dist = int(D[0][i])
        # Calculate similarity percentage (64 bits - 0 different bits = 100% similar)
        similarity_pct = 100 * (64 - hamming_dist) / 64
        
        results.append({
            "id": frame_id,
            "distance": hamming_dist,
            "similarity_percentage": similarity_pct,
            "metadata": metadata.get(frame_id, {})
        })
    t_end_results = time.time()
    results_time = t_end_results - t_start_results
    
    # Total time
    total_time = time.time() - start_time
    
    timings = {
        "processing_time": proc_time,
        "search_time": search_time,
        "results_time": results_time,
        "total_time": total_time
    }
    
    return results, timings

# === API Routes ===

@app.on_event("startup")
async def startup_load_index():
    """Load FAISS index and metadata on startup"""
    global index, metadata, binary_vectors
    
    try:
        logger.info(f"Memory before loading data: {update_max_memory():.2f} MB")
        
        # Load raw hashes
        if not os.path.exists(PHASHES_PATH):
            logger.error(f"Hash file not found: {PHASHES_PATH}")
            raise FileNotFoundError(f"Hash file not found: {PHASHES_PATH}")
            
        hashes_raw = np.load(PHASHES_PATH)
        logger.info(f"Loaded {len(hashes_raw)} hashes of 64 bits")
        logger.info(f"Memory after loading hashes: {update_max_memory():.2f} MB")
        
        # Load metadata
        if not os.path.exists(METADATA_PATH):
            logger.error(f"Metadata file not found: {METADATA_PATH}")
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")
            
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded metadata with {len(metadata)} entries")
        logger.info(f"Memory after loading metadata: {update_max_memory():.2f} MB")
        
        # Prepare binary vectors for FAISS
        logger.info("Preparing hashes for binary index...")
        binary_vectors = np.zeros((len(hashes_raw), 8), dtype=np.uint8)
        for i, hash_value in enumerate(hashes_raw):
            binary_vectors[i] = np.array([hash_value], dtype=np.uint64).view(np.uint8)
        
        # Create FAISS binary index with Hamming distance
        logger.info("Creating FAISS binary index with Hamming distance...")
        binary_dim = 64  # 64 bits per hash
        base_index = faiss.IndexBinaryFlat(binary_dim)
        index = faiss.IndexBinaryIDMap(base_index)
        ids = np.array(list(metadata.keys()), dtype=np.int64)
        index.add_with_ids(binary_vectors, ids)
        
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        logger.info(f"Memory after loading data: {update_max_memory():.2f} MB")
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "name": "WhatsEpisode API",
        "version": "1.0.0", 
        "description": "API for finding similar anime frames using perceptual hash similarity",
        "endpoints": {
            "search": "POST /search",
            "status": "GET /status",
            "health": "GET /health"
        }
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API status and data statistics"""
    global index, metadata, max_memory
    
    is_data_loaded = index is not None and metadata is not None
    
    return StatusResponse(
        status="ok",
        data_loaded=is_data_loaded,
        index_size=index.ntotal if is_data_loaded else 0,
        metadata_count=len(metadata) if is_data_loaded else 0,
        memory_usage_mb=get_memory_usage(),
        max_memory_usage_mb=max_memory
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/search", response_model=SearchResponse)
async def search_frame(
    file: UploadFile = File(...),
    max_results: int = Query(5, description="Number of results to return", ge=1, le=100)
):
    """
    Search for similar anime frames using an uploaded image
    
    Args:
        file: Image file to search for
        max_results: Maximum number of results to return (1-100)
        
    Returns:
        Search results with metadata and similarity scores
    """
    global index, metadata
    
    if index is None or metadata is None:
        raise HTTPException(
            status_code=500, 
            detail="Search index not loaded. The API may still be initializing."
        )
    
    try:
        # Validate file is an image
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image, got {file.content_type}"
            )
        
        # Read the image
        image_bytes = await file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image. Make sure it's a valid image file."
            )
        
        # Search for similar frames
        results, timings = search_similar_frame(img, k=max_results)
        
        # Format results for response model
        formatted_results = []
        for r in results:
            metadata = r["metadata"]
            
            # Format time as MM:SS or HH:MM:SS if hours > 0
            second = metadata.get("second", 0)
            if second is not None:
                minutes = second // 60
                hours = minutes // 60
                if hours > 0:
                    # Format as HH:MM:SS
                    metadata["time_formatted"] = f"{hours}:{minutes % 60:02d}:{second % 60:02d}"
                else:
                    # Format as MM:SS
                    metadata["time_formatted"] = f"{minutes}:{second % 60:02d}"
            else:
                metadata["time_formatted"] = "0:00"
                
            formatted_results.append(
                SearchResult(
                    id=r["id"],
                    distance=r["distance"],
                    similarity_percentage=r["similarity_percentage"],
                    metadata=metadata
                )
            )
        
        # Create search timings object
        search_timings = SearchTimings(
            processing_time=timings["processing_time"],
            search_time=timings["search_time"],
            results_time=timings["results_time"],
            total_time=timings["total_time"]
        )
        
        # Return the response
        return SearchResponse(
            results=formatted_results,
            timings=search_timings,
            total_results=len(formatted_results),
            query_image=file.filename or "uploaded_image"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )