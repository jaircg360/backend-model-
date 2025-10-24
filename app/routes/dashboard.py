"""
Endpoints para el Dashboard
"""

from fastapi import APIRouter, HTTPException
import os
from pathlib import Path
import logging

from app.services.supabase_service import SupabaseService
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_dashboard_stats():
    """
    Obtiene estadísticas generales para el dashboard
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_dashboard_stats()
        
        if result["success"]:
            stats = result["data"]
            
            # Calcular tamaño de almacenamiento
            storage_used_mb = calculate_storage_usage()
            stats["storage_used_mb"] = storage_used_mb
            
            return {
                "success": True,
                "stats": stats
            }
        else:
            # Modo local
            local_stats = get_local_stats()
            return {
                "success": True,
                "stats": local_stats,
                "message": result["message"]
            }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener estadísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent-activity")
async def get_recent_activity():
    """
    Obtiene la actividad reciente del sistema
    """
    try:
        supabase_service = SupabaseService()
        
        # Obtener datasets recientes
        datasets_result = supabase_service.get_all_datasets(limit=10)
        
        # Obtener modelos recientes
        models_result = supabase_service.get_all_models(limit=10)
        
        activities = []
        
        # Combinar y ordenar por fecha
        if datasets_result["success"] and datasets_result["data"]:
            for dataset in datasets_result["data"]:
                activities.append({
                    "type": "dataset",
                    "action": "uploaded" if dataset.get("status") == "uploaded" else "updated",
                    "name": dataset.get("file_name"),
                    "id": dataset.get("id"),
                    "timestamp": dataset.get("created_at"),
                    "details": {
                        "rows": dataset.get("rows"),
                        "columns": dataset.get("columns"),
                        "status": dataset.get("status")
                    }
                })
        
        if models_result["success"] and models_result["data"]:
            for model in models_result["data"]:
                activities.append({
                    "type": "model",
                    "action": "trained",
                    "name": model.get("model_name"),
                    "id": model.get("id"),
                    "timestamp": model.get("created_at"),
                    "details": {
                        "model_type": model.get("model_type"),
                        "metrics": model.get("metrics")
                    }
                })
        
        # Ordenar por timestamp
        activities.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "success": True,
            "activities": activities[:20]  # Últimas 20 actividades
        }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener actividad reciente: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets-summary")
async def get_datasets_summary():
    """
    Obtiene un resumen de todos los datasets
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_all_datasets()
        
        if not result["success"]:
            return {
                "success": True,
                "summary": [],
                "message": result["message"]
            }
        
        datasets = result["data"] or []
        
        summary = []
        for dataset in datasets:
            summary.append({
                "id": dataset.get("id"),
                "name": dataset.get("file_name"),
                "rows": dataset.get("rows"),
                "columns": dataset.get("columns"),
                "status": dataset.get("status"),
                "created_at": dataset.get("created_at"),
                "has_missing_values": any(
                    count > 0 
                    for count in (dataset.get("missing_values") or {}).values()
                ),
                "missing_percentage": sum(
                    (dataset.get("missing_values") or {}).values()
                ) / (dataset.get("rows", 1) * dataset.get("columns", 1)) * 100 if dataset.get("rows") else 0
            })
        
        return {
            "success": True,
            "summary": summary,
            "total": len(summary)
        }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener resumen de datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models-summary")
async def get_models_summary():
    """
    Obtiene un resumen de todos los modelos
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_all_models()
        
        if not result["success"]:
            return {
                "success": True,
                "summary": [],
                "message": result["message"]
            }
        
        models = result["data"] or []
        
        summary = []
        for model in models:
            metrics = model.get("metrics") or {}
            
            # Determinar métrica principal según tipo
            main_metric = None
            model_type = model.get("model_type", "")
            
            if "regression" in model_type.lower():
                main_metric = {"name": "R² Score", "value": metrics.get("r2_score")}
            elif "classification" in model_type.lower():
                main_metric = {"name": "Accuracy", "value": metrics.get("accuracy")}
            else:
                # Tomar la primera métrica disponible
                if metrics:
                    first_key = list(metrics.keys())[0]
                    main_metric = {"name": first_key, "value": metrics[first_key]}
            
            summary.append({
                "id": model.get("id"),
                "name": model.get("model_name"),
                "type": model.get("model_type"),
                "dataset_id": model.get("dataset_id"),
                "created_at": model.get("created_at"),
                "main_metric": main_metric,
                "all_metrics": metrics
            })
        
        return {
            "success": True,
            "summary": summary,
            "total": len(summary)
        }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener resumen de modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-info")
async def get_system_info():
    """
    Obtiene información del sistema
    """
    try:
        import platform
        import sys
        
        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "storage": {
                "uploads_dir": settings.UPLOADS_DIR,
                "models_dir": settings.MODELS_DIR,
                "exports_dir": settings.EXPORTS_DIR,
                "total_size_mb": calculate_storage_usage()
            },
            "supabase": {
                "configured": bool(settings.SUPABASE_URL and settings.SUPABASE_KEY),
                "url": settings.SUPABASE_URL if settings.SUPABASE_URL else "Not configured"
            },
            "limits": {
                "max_file_size_mb": settings.MAX_FILE_SIZE / (1024 * 1024),
                "allowed_extensions": list(settings.ALLOWED_EXTENSIONS)
            }
        }
        
        return {
            "success": True,
            "info": info
        }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener info del sistema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= HELPER FUNCTIONS =============

def calculate_storage_usage() -> float:
    """Calcula el uso de almacenamiento en MB"""
    total_size = 0
    
    directories = [
        settings.UPLOADS_DIR,
        settings.MODELS_DIR,
        settings.EXPORTS_DIR
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
    
    return round(total_size / (1024 * 1024), 2)  # Convertir a MB

def get_local_stats() -> dict:
    """Obtiene estadísticas locales cuando Supabase no está disponible"""
    uploads_dir = Path(settings.UPLOADS_DIR)
    models_dir = Path(settings.MODELS_DIR)
    
    total_datasets = 0
    total_models = 0
    
    if uploads_dir.exists():
        total_datasets = len(list(uploads_dir.glob("*.*")))
    
    if models_dir.exists():
        total_models = len(list(models_dir.glob("*.*")))
    
    return {
        "total_datasets": total_datasets,
        "total_models": total_models,
        "total_cleaned_datasets": 0,
        "recent_datasets": [],
        "recent_models": [],
        "storage_used_mb": calculate_storage_usage()
    }

