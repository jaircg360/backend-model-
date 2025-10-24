"""
Endpoints para limpieza de datos
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
import logging

from app.services.data_cleaner import DataCleanerService
from app.services.supabase_service import SupabaseService
from app.services.agent_service import AgentService
from app.models.schemas import CleaningRequest, CleaningResponse
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=CleaningResponse)
async def clean_dataset(request: CleaningRequest):
    """
    Limpia un dataset aplicando las acciones especificadas
    """
    try:
        # Buscar archivo
        supabase_service = SupabaseService()
        dataset_result = supabase_service.get_dataset(request.file_id)
        
        file_path = None
        if dataset_result["success"] and dataset_result["data"]:
            file_path = dataset_result["data"]["file_path"]
        else:
            # Buscar archivo local
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{request.file_id}.*"))
            if matching_files:
                file_path = str(matching_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Cargar y limpiar datos
        cleaner = DataCleanerService()
        cleaner.load_data(file_path)
        
        logger.info(f"🧹 Limpiando dataset {request.file_id}...")
        
        # Aplicar pipeline de limpieza
        results = cleaner.apply_cleaning_pipeline(
            actions=request.actions,
            encoding_method=request.encoding_method,
            outlier_method=request.outlier_method
        )
        
        # Guardar datos limpios
        cleaned_file_name = f"{request.file_id}_cleaned{Path(file_path).suffix}"
        cleaned_file_path = os.path.join(settings.UPLOADS_DIR, cleaned_file_name)
        cleaner.save_cleaned_data(cleaned_file_path)
        
        # Actualizar en Supabase
        supabase_service.update_dataset(request.file_id, {
            "status": "cleaned",
            "rows": results["final_shape"][0],
            "file_path": cleaned_file_path
        })
        
        # Crear log de limpieza
        log_data = {
            "dataset_id": request.file_id,
            "actions": [action.value for action in request.actions],
            "original_rows": results["original_shape"][0],
            "cleaned_rows": results["final_shape"][0],
            "statistics": results["summary"]
        }
        supabase_service.create_cleaning_log(log_data)
        
        # Preview de datos limpios
        preview = cleaner.get_preview(n=10)
        
        logger.info(f"✅ Limpieza completada: {results['original_shape'][0]} → {results['final_shape'][0]} filas")
        
        return CleaningResponse(
            success=True,
            message="Dataset limpiado exitosamente",
            file_id=request.file_id,
            original_rows=results["original_shape"][0],
            cleaned_rows=results["final_shape"][0],
            actions_applied=results["applied_actions"],
            preview=preview,
            statistics={
                "original_shape": results["original_shape"],
                "final_shape": results["final_shape"],
                "actions_results": results["actions_results"],
                "summary": results["summary"]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al limpiar dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al limpiar datos: {str(e)}")

@router.get("/recommendations/{file_id}")
async def get_cleaning_recommendations(file_id: str):
    """
    Obtiene recomendaciones de limpieza para un dataset
    """
    try:
        # Buscar archivo
        supabase_service = SupabaseService()
        dataset_result = supabase_service.get_dataset(file_id)
        
        file_path = None
        if dataset_result["success"] and dataset_result["data"]:
            file_path = dataset_result["data"]["file_path"]
        else:
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{file_id}.*"))
            if matching_files:
                file_path = str(matching_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Analizar con el agente
        agent = AgentService()
        analysis = agent.analyze_dataset(file_path)
        
        if not analysis["success"]:
            raise HTTPException(status_code=500, detail=analysis.get("message", "Error al analizar"))
        
        recommendations = agent.get_cleaning_recommendations(analysis["analysis"])
        
        return {
            "success": True,
            "file_id": file_id,
            "analysis": analysis["analysis"],
            "recommendations": recommendations,
            "suggestions": analysis.get("suggestions", []),
            "warnings": analysis.get("warnings"),
            "next_step": analysis.get("next_step")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al obtener recomendaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{file_id}")
async def get_cleaning_history(file_id: str):
    """
    Obtiene el historial de limpieza de un dataset
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_cleaning_logs_by_dataset(file_id)
        
        if result["success"]:
            return {
                "success": True,
                "file_id": file_id,
                "history": result["data"] or []
            }
        else:
            return {
                "success": True,
                "file_id": file_id,
                "history": [],
                "message": result["message"]
            }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener historial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preview/{file_id}")
async def preview_cleaning(file_id: str, request: CleaningRequest):
    """
    Previsualiza el efecto de las acciones de limpieza sin guardar
    """
    try:
        # Buscar archivo
        supabase_service = SupabaseService()
        dataset_result = supabase_service.get_dataset(file_id)
        
        file_path = None
        if dataset_result["success"] and dataset_result["data"]:
            file_path = dataset_result["data"]["file_path"]
        else:
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{file_id}.*"))
            if matching_files:
                file_path = str(matching_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Cargar y limpiar datos (sin guardar)
        cleaner = DataCleanerService()
        cleaner.load_data(file_path)
        
        original_preview = cleaner.get_preview(n=5)
        original_summary = cleaner.get_data_summary()
        
        # Aplicar limpieza
        results = cleaner.apply_cleaning_pipeline(
            actions=request.actions,
            encoding_method=request.encoding_method,
            outlier_method=request.outlier_method
        )
        
        cleaned_preview = cleaner.get_preview(n=5)
        
        return {
            "success": True,
            "original": {
                "shape": original_summary["shape"],
                "preview": original_preview,
                "missing_values": original_summary["missing_values"],
                "duplicates": original_summary["duplicates"]
            },
            "cleaned": {
                "shape": results["final_shape"],
                "preview": cleaned_preview,
                "missing_values": results["summary"]["missing_values"],
                "duplicates": results["summary"]["duplicates"]
            },
            "changes": {
                "rows_removed": results["original_shape"][0] - results["final_shape"][0],
                "columns_changed": results["final_shape"][1] - results["original_shape"][1],
                "actions_applied": results["applied_actions"]
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en preview de limpieza: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

