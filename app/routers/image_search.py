from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path

from app.models.image_search_models import SearchResponse, ErrorResponse
from app.services.image_search_service import ImageSearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_anime_episode(
    image: UploadFile = File(..., description="Image to search for anime episode")
):
    """
    Search for anime episode using an image.
    
    - **image**: Image file (JPG, PNG, BMP, TIFF, WEBP)
    
    Returns information about the found episode, including:
    - Anime name
    - Season and episode
    - Timestamp in the episode
    - Search similarity
    - Video preview URL (if available)
    """
    
    # Validate file type
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be a valid image"
        )
    
    # Create temporary file
    temp_file = None
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename).suffix) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Validate if it's a valid image
        if not ImageSearchService.validate_image_file(temp_file_path):
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format"
            )
        
        # Perform search
        search_result = ImageSearchService.search_anime_episode(temp_file_path)
        
        return SearchResponse(**search_result)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


@router.get("/health")
async def health_check():
    """
    Check if the search service is working.
    """
    try:
        # Get service status
        status = ImageSearchService.get_service_status()
        
        if not status["initialized"]:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Service not initialized"}
            )
        
        if not status["index_loaded"]:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "FAISS index not loaded"}
            )
        
        if not status["metadata_loaded"]:
            return JSONResponse(
                status_code=503, 
                content={"status": "unhealthy", "reason": "Metadata not loaded"}
            )
        
        return {
            "status": "healthy", 
            "message": "Search service operational",
            "stats": {
                "index_size": status["index_size"],
                "metadata_entries": status["metadata_entries"]
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": f"Error: {str(e)}"}
        )


@router.get("/stats")
async def get_service_stats():
    """
    Get detailed service statistics and information.
    """
    try:
        status = ImageSearchService.get_service_status()
        
        return {
            "service": "AniFind Search Service",
            "version": "1.0.0",
            "status": status,
            "endpoints": {
                "search": "/api/v1/search",
                "health": "/api/v1/health",
                "stats": "/api/v1/stats"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
