# main.py

from typing import Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import os
import tempfile

# Importar nosso serviço (como import no Node.js)
from app.services import FrameExtractorService, FrameSearchService

app = FastAPI(
    title="Anime Frame Finder API",
    description="API para identificar frames de anime",
    version="1.0.0"
)

# Instanciar os serviços (como new Service() no Node)
frame_service = FrameExtractorService()
search_service = FrameSearchService()

@app.get("/")
def read_root():
    return {
        "message": "Anime Frame Finder API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /extract-frames",
            "search": "POST /search-frame",
            "list": "GET /extractions",
            "stats": "GET /stats",
            "clear": "DELETE /extractions"
        }
    }

@app.post("/extract-frames")
async def extract_frames(
    file: UploadFile = File(...),
    anime_name: str = Form(...),
    episode: int = Form(...)
):
    """
    Endpoint para extrair frames de um vídeo
    """
    try:
        # Validar arquivo
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser um vídeo")
        
        # Salvar arquivo temporariamente (como multer no Node.js)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Processar vídeo
            result = frame_service.extract_frames_from_video(
                video_path=temp_path,
                anime_name=anime_name,
                episode=episode
            )
            
            return result
            
        finally:
            # Limpar arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar vídeo: {str(e)}")

@app.post("/search-frame")
async def search_frame(
    file: UploadFile = File(...),
    threshold: int = Form(10)
):
    """
    Buscar frames similares a partir de uma imagem enviada
    """
    try:
        # Validar arquivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        # Ler imagem
        image_data = await file.read()
        
        # Buscar frames similares
        result = search_service.search_by_upload(image_data, threshold)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na busca: {str(e)}")

@app.get("/stats")
def get_stats():
    """
    Estatísticas do banco de frames
    """
    return search_service.get_stats()

@app.get("/extractions")
def get_extractions():
    """
    Listar todas as extrações salvas
    """
    return frame_service.get_all_extractions()

@app.delete("/extractions")
def clear_extractions():
    """
    Limpar todas as extrações (desenvolvimento)
    """
    return frame_service.clear_extractions()

@app.get("/health")
def health_check():
    """
    Health check
    """
    data = frame_service.get_all_extractions()
    total_extractions = len(data.get("extractions", []))
    
    return {
        "status": "healthy",
        "total_extractions": total_extractions
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )