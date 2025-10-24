"""
Endpoints para subir y gestionar archivos
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import uuid
import logging

from app.services.data_cleaner import DataCleanerService
from app.services.supabase_service import SupabaseService
from app.services.agent_service import AgentService
from app.models.schemas import UploadResponse
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Sube un archivo CSV o Excel para análisis
    """
    try:
        # Validar extensión
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado. Usa: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Generar ID único
        file_id = str(uuid.uuid4())
        file_name = f"{file_id}{file_extension}"
        file_path = os.path.join(settings.UPLOADS_DIR, file_name)
        
        # Guardar archivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"📤 Archivo subido: {file.filename} -> {file_path}")
        
        # Analizar archivo
        cleaner = DataCleanerService()
        df = cleaner.load_data(file_path)
        summary = cleaner.get_data_summary()
        preview = cleaner.get_preview(n=5)
        
        # Guardar en Supabase
        supabase_service = SupabaseService()
        dataset_data = {
            "file_name": file.filename,
            "file_path": file_path,
            "status": "uploaded",
            "rows": summary["shape"][0],
            "columns": summary["shape"][1],
            "column_types": summary["dtypes"],
            "missing_values": summary["missing_values"]
        }
        
        db_result = supabase_service.create_dataset(dataset_data)
        
        # Obtener ID del dataset (puede ser del resultado de Supabase o generado localmente)
        if db_result["success"] and db_result["data"]:
            file_id = db_result["data"]["id"]
        
        # Análisis del agente
        agent = AgentService()
        agent_analysis = agent.analyze_dataset(file_path)
        
        return UploadResponse(
            success=True,
            message=f"Archivo '{file.filename}' subido exitosamente",
            file_id=file_id,
            file_name=file.filename,
            rows=summary["shape"][0],
            columns=summary["shape"][1],
            preview=preview,
            column_info={
                "dtypes": summary["dtypes"],
                "missing_values": summary["missing_values"],
                "missing_percentage": summary["missing_percentage"],
                "numeric_columns": summary["numeric_columns"],
                "categorical_columns": summary["categorical_columns"],
                "duplicates": summary["duplicates"],
                "agent_suggestions": agent_analysis.get("suggestions", []),
                "warnings": agent_analysis.get("warnings")
            }
        )
    
    except Exception as e:
        logger.error(f"❌ Error al subir archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar archivo: {str(e)}")

@router.get("/list")
async def list_uploads():
    """
    Lista todos los archivos subidos
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_all_datasets()
        
        if result["success"]:
            return {
                "success": True,
                "datasets": result["data"] or []
            }
        else:
            # Si Supabase no está disponible, listar archivos locales
            uploads_dir = Path(settings.UPLOADS_DIR)
            if not uploads_dir.exists():
                return {"success": True, "datasets": []}
            
            files = []
            cleaner = DataCleanerService()
            
            for file_path in uploads_dir.glob("*.*"):
                if file_path.suffix.lower() in settings.ALLOWED_EXTENSIONS:
                    try:
                        # Leer info básica del archivo
                        df = cleaner.load_data(str(file_path))
                        summary = cleaner.get_data_summary()
                        
                        files.append({
                            "id": file_path.stem,
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "status": "uploaded",
                            "rows": summary["shape"][0],
                            "columns": summary["shape"][1],
                            "column_types": summary["dtypes"],
                            "missing_values": summary["missing_values"],
                            "created_at": file_path.stat().st_ctime,
                            "updated_at": file_path.stat().st_mtime
                        })
                    except Exception as e:
                        logger.warning(f"No se pudo leer {file_path.name}: {str(e)}")
                        # Agregar sin metadatos si falla la lectura
                        files.append({
                            "id": file_path.stem,
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "status": "uploaded",
                            "rows": 0,
                            "columns": 0
                        })
            
            return {
                "success": True,
                "datasets": files,
                "message": result["message"]
            }
    
    except Exception as e:
        logger.error(f"❌ Error al listar archivos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}")
async def get_file_info(file_id: str):
    """
    Obtiene información de un archivo específico
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_dataset(file_id)
        
        if result["success"] and result["data"]:
            dataset = result["data"]
            
            # Cargar preview
            cleaner = DataCleanerService()
            cleaner.load_data(dataset["file_path"])
            preview = cleaner.get_preview(n=10)
            
            return {
                "success": True,
                "dataset": dataset,
                "preview": preview
            }
        else:
            # Buscar archivo local
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{file_id}.*"))
            
            if not matching_files:
                raise HTTPException(status_code=404, detail="Archivo no encontrado")
            
            file_path = matching_files[0]
            cleaner = DataCleanerService()
            cleaner.load_data(str(file_path))
            preview = cleaner.get_preview(n=10)
            summary = cleaner.get_data_summary()
            
            return {
                "success": True,
                "dataset": {
                    "id": file_id,
                    "file_id": file_id,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "status": "uploaded",
                    "rows": summary["shape"][0],
                    "columns": summary["shape"][1],
                    "column_types": summary["dtypes"],
                    "missing_values": summary["missing_values"],
                    "created_at": file_path.stat().st_ctime,
                    "updated_at": file_path.stat().st_mtime
                },
                "preview": preview
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al obtener archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    Elimina un archivo
    """
    try:
        supabase_service = SupabaseService()
        
        # Obtener info del dataset
        dataset_result = supabase_service.get_dataset(file_id)
        
        if dataset_result["success"] and dataset_result["data"]:
            file_path = dataset_result["data"]["file_path"]
            
            # Eliminar archivo físico
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ Archivo eliminado: {file_path}")
            
            # Eliminar de BD
            supabase_service.delete_dataset(file_id)
        else:
            # Buscar y eliminar archivo local
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{file_id}.*"))
            
            if matching_files:
                matching_files[0].unlink()
                logger.info(f"🗑️ Archivo local eliminado: {matching_files[0]}")
        
        return {
            "success": True,
            "message": "Archivo eliminado exitosamente"
        }
    
    except Exception as e:
        logger.error(f"❌ Error al eliminar archivo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

