# app/models/video_models.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class VideoUploadRequest(BaseModel):
    """
    Modelo para request de upload de vídeo
    Similar a uma interface TypeScript
    """
    anime_name: str
    episode: int

class FrameData(BaseModel):
    """
    Dados de um frame extraído
    """
    filename: str
    filepath: str
    frame_number: int
    second: int
    position: str  # "start", "middle", "end"
    timestamp: str  # "HH:MM:SS"
    anime: str
    episode: int

class VideoInfo(BaseModel):
    """
    Informações do vídeo
    """
    fps: float
    duration: float
    total_frames: int

class ExtractionResult(BaseModel):
    """
    Resultado da extração de frames
    """
    anime: str
    episode: int
    video_info: VideoInfo
    extracted_frames: List[FrameData]
    extraction_date: str

class ApiResponse(BaseModel):
    """
    Resposta padrão da API
    """
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None