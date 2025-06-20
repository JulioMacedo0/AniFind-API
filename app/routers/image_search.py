"""
Image search API routes for anime frame similarity search

This module implements a FastAPI router for image search based on init.py logic
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
import time
import io
from typing import List, Dict, Any, Optional

# Import the models
from app.models.image_search_models import SearchResponse, SearchResultItem, SearchTimings, StatusResponse, FrameMetadata

# Import the service
from app.services.image_search_service import ImageSearchService

# Create a router
router = APIRouter(
    prefix="/image-search",
    tags=["image-search"],
    responses={404: {"description": "Not found"}},
)

# Setup logging
logger = logging.getLogger("image_search_routes")

# Create the service instance
service = ImageSearchService()

@router.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get the status of the image search service
    
    Returns:
        Status information including memory usage and data info
    """
    status = service.get_status()
    return StatusResponse(
        status="ok",
        data_loaded=status["data_loaded"],
        index_size=status["index_size"],
        metadata_count=status["metadata_count"],
        memory_usage_mb=status["current_memory_mb"],
        max_memory_usage_mb=status["max_memory_mb"]
    )

@router.post("/search", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Query(5, description="Number of results to return", gt=0, le=100)
):
    """
    Search for similar anime frames using an uploaded image
    
    Args:
        file: The image file to search for
        top_k: Number of results to return (default: 5)
        
    Returns:
        Search results with similarity information and metadata
    """
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got: {file.content_type}"
        )
    
    try:
        # Read file contents
        image_bytes = await file.read()
        
        # Perform the search
        results, timings = service.search_from_bytes(image_bytes, k=top_k)
        
        # Format the results
        formatted_results = []
        for result in results:
            # Get the metadata
            metadata = result["metadata"] or {}
            
            # Format the time
            second = metadata.get("second", 0)
            
            # Format time as MM:SS or HH:MM:SS if hours > 0
            if second is not None:
                minutes = second // 60
                hours = minutes // 60
                if hours > 0:
                    # Format as HH:MM:SS
                    time_formatted = f"{hours}:{minutes % 60:02d}:{second % 60:02d}"
                else:
                    # Format as MM:SS
                    time_formatted = f"{minutes}:{second % 60:02d}"
            else:
                time_formatted = "0:00"
            
            # Create formatted metadata
            formatted_metadata = FrameMetadata(
                anime=metadata.get("anime", "Unknown"),
                season=metadata.get("season", 1),
                episode=metadata.get("episode", 1),
                second=second,
                position=metadata.get("position", ""),
                filepath=metadata.get("filepath", ""),
                time_formatted=time_formatted
            )
            
            # Add to results
            formatted_results.append(SearchResultItem(
                id=result["id"],
                distance=result["distance"],
                similarity_percentage=result["similarity_percentage"],
                metadata=metadata,
                formatted_metadata=formatted_metadata
            ))
        
        # Create timing info
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
        
    except Exception as e:
        logger.error(f"Error during image search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during image search: {str(e)}"
        )
