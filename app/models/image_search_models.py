from pydantic import BaseModel
from typing import List, Optional


class SearchResult(BaseModel):
    rank: int
    anime: str
    season: int
    episode: int
    timecode: str
    second: float
    similarity: float
    anime_id: int  
    source_file: str
    preview_source_path: str
    preview_video: Optional[str] = None


class SearchResponse(BaseModel):
    top_result: Optional[SearchResult]
    all_results: List[SearchResult]
    preview_url: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
