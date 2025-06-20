"""
Pydantic models for the image search API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class FrameMetadata(BaseModel):
    """Metadata for a frame from an anime episode"""
    anime: str = Field(default="Unknown", description="Anime name")
    season: int = Field(default=1, description="Season number")
    episode: int = Field(default=1, description="Episode number")
    second: int = Field(default=0, description="Second in the episode")
    position: str = Field(default="", description="Position in the second (start, middle, end)")
    filepath: str = Field(default="", description="Path to the source video file")
    time_formatted: str = Field(default="0:00", description="Formatted time (HH:MM:SS or MM:SS)")

class SearchResultItem(BaseModel):
    """An individual search result item"""
    id: int = Field(..., description="ID of the result in the database")
    distance: int = Field(..., description="Hamming distance (number of different bits, 0-64)")
    similarity_percentage: float = Field(..., description="Similarity percentage (100% = identical)")
    metadata: Dict[str, Any] = Field(..., description="Original metadata for the frame")
    formatted_metadata: Optional[FrameMetadata] = Field(None, description="Formatted metadata")

class SearchTimings(BaseModel):
    """Timing information for the search process"""
    processing_time: float = Field(..., description="Time to process the query image (seconds)")
    search_time: float = Field(..., description="Time to search in the index (seconds)")
    results_time: float = Field(..., description="Time to format the results (seconds)")
    total_time: float = Field(..., description="Total search time (seconds)")

class SearchResponse(BaseModel):
    """Response for the image search endpoint"""
    results: List[SearchResultItem] = Field(..., description="Search results ordered by similarity")
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
