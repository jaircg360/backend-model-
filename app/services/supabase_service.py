"""
Servicio para interactuar con Supabase con manejo robusto de errores
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

from app.database import get_supabase_client, safe_db_operation
from app.models.schemas import DatasetDB, ModelDB, CleaningLogDB

logger = logging.getLogger(__name__)

class SupabaseService:
    """Servicio para todas las operaciones de Supabase"""

    def __init__(self):
        self.client = get_supabase_client()

    def is_available(self) -> bool:
        """Verifica si Supabase está disponible"""
        return self.client is not None
    
    # ============= DATASETS =============
    
    def create_dataset(self, dataset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un nuevo registro de dataset"""
        def operation(client):
            data = {
                "id": str(uuid.uuid4()),
                "file_name": dataset_data.get("file_name"),
                "file_path": dataset_data.get("file_path"),
                "status": dataset_data.get("status", "uploaded"),
                "rows": dataset_data.get("rows"),
                "columns": dataset_data.get("columns"),
                "column_types": dataset_data.get("column_types"),
                "missing_values": dataset_data.get("missing_values"),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            result = client.table("datasets").insert(data).execute()
            logger.info(f"✅ Dataset creado en Supabase: {data['id']}")
            return result.data[0] if result.data else data
        
        return safe_db_operation(
            operation,
            "datasets",
            "Error al crear dataset en Supabase"
        )
    
    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Obtiene un dataset por ID"""
        def operation(client):
            result = client.table("datasets").select("*").eq("id", dataset_id).execute()
            
            if not result.data:
                raise ValueError(f"Dataset {dataset_id} no encontrado")
            
            return result.data[0]
        
        return safe_db_operation(
            operation,
            "datasets",
            f"Error al obtener dataset {dataset_id}"
        )
    
    def get_all_datasets(self, limit: int = 100) -> Dict[str, Any]:
        """Obtiene todos los datasets"""
        def operation(client):
            result = client.table("datasets").select("*").order("created_at", desc=True).limit(limit).execute()
            return result.data
        
        return safe_db_operation(
            operation,
            "datasets",
            "Error al obtener datasets"
        )
    
    def update_dataset(self, dataset_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Actualiza un dataset"""
        def operation(client):
            updates["updated_at"] = datetime.now().isoformat()
            result = client.table("datasets").update(updates).eq("id", dataset_id).execute()
            
            if result.data:
                logger.info(f"✅ Dataset actualizado: {dataset_id}")
                return result.data[0]
            return None
        
        return safe_db_operation(
            operation,
            "datasets",
            f"Error al actualizar dataset {dataset_id}"
        )
    
    def delete_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Elimina un dataset"""
        def operation(client):
            result = client.table("datasets").delete().eq("id", dataset_id).execute()
            logger.info(f"🗑️ Dataset eliminado: {dataset_id}")
            return {"deleted": True, "id": dataset_id}
        
        return safe_db_operation(
            operation,
            "datasets",
            f"Error al eliminar dataset {dataset_id}"
        )
    
    # ============= MODELS =============
    
    def create_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un nuevo registro de modelo"""
        def operation(client):
            data = {
                "id": str(uuid.uuid4()),
                "model_name": model_data.get("model_name"),
                "model_type": model_data.get("model_type"),
                "dataset_id": model_data.get("dataset_id"),
                "model_path": model_data.get("model_path"),
                "metrics": model_data.get("metrics"),
                "feature_importance": model_data.get("feature_importance"),
                "hyperparameters": model_data.get("hyperparameters"),
                "created_at": datetime.now().isoformat(),
                "status": model_data.get("status", "trained")
            }
            
            result = client.table("models").insert(data).execute()
            logger.info(f"✅ Modelo creado en Supabase: {data['id']}")
            return result.data[0] if result.data else data
        
        return safe_db_operation(
            operation,
            "models",
            "Error al crear modelo en Supabase"
        )
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Obtiene un modelo por ID"""
        def operation(client):
            result = client.table("models").select("*").eq("id", model_id).execute()
            
            if not result.data:
                raise ValueError(f"Modelo {model_id} no encontrado")
            
            return result.data[0]
        
        return safe_db_operation(
            operation,
            "models",
            f"Error al obtener modelo {model_id}"
        )
    
    def get_models_by_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Obtiene todos los modelos de un dataset"""
        def operation(client):
            result = client.table("models").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).execute()
            return result.data
        
        return safe_db_operation(
            operation,
            "models",
            f"Error al obtener modelos del dataset {dataset_id}"
        )
    
    def get_all_models(self, limit: int = 100) -> Dict[str, Any]:
        """Obtiene todos los modelos"""
        def operation(client):
            result = client.table("models").select("*").order("created_at", desc=True).limit(limit).execute()
            return result.data
        
        return safe_db_operation(
            operation,
            "models",
            "Error al obtener modelos"
        )
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Elimina un modelo"""
        def operation(client):
            result = client.table("models").delete().eq("id", model_id).execute()
            logger.info(f"🗑️ Modelo eliminado: {model_id}")
            return {"deleted": True, "id": model_id}
        
        return safe_db_operation(
            operation,
            "models",
            f"Error al eliminar modelo {model_id}"
        )
    
    # ============= CLEANING LOGS =============
    
    def create_cleaning_log(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea un log de limpieza"""
        def operation(client):
            data = {
                "id": str(uuid.uuid4()),
                "dataset_id": log_data.get("dataset_id"),
                "actions": log_data.get("actions"),
                "original_rows": log_data.get("original_rows"),
                "cleaned_rows": log_data.get("cleaned_rows"),
                "statistics": log_data.get("statistics"),
                "created_at": datetime.now().isoformat()
            }
            
            result = client.table("cleaning_logs").insert(data).execute()
            logger.info(f"✅ Log de limpieza creado: {data['id']}")
            return result.data[0] if result.data else data
        
        return safe_db_operation(
            operation,
            "cleaning_logs",
            "Error al crear log de limpieza"
        )
    
    def get_cleaning_logs_by_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Obtiene todos los logs de limpieza de un dataset"""
        def operation(client):
            result = client.table("cleaning_logs").select("*").eq("dataset_id", dataset_id).order("created_at", desc=True).execute()
            return result.data
        
        return safe_db_operation(
            operation,
            "cleaning_logs",
            f"Error al obtener logs del dataset {dataset_id}"
        )
    
    # ============= DASHBOARD STATS =============
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas para el dashboard"""
        def operation(client):
            # Total datasets
            datasets_result = client.table("datasets").select("id", count="exact").execute()
            total_datasets = datasets_result.count if hasattr(datasets_result, 'count') else len(datasets_result.data)
            
            # Total modelos
            models_result = client.table("models").select("id", count="exact").execute()
            total_models = models_result.count if hasattr(models_result, 'count') else len(models_result.data)
            
            # Datasets limpios
            cleaned_result = client.table("datasets").select("id", count="exact").eq("status", "cleaned").execute()
            total_cleaned = cleaned_result.count if hasattr(cleaned_result, 'count') else len(cleaned_result.data)
            
            # Datasets recientes
            recent_datasets = client.table("datasets").select("*").order("created_at", desc=True).limit(5).execute()
            
            # Modelos recientes
            recent_models = client.table("models").select("*").order("created_at", desc=True).limit(5).execute()
            
            return {
                "total_datasets": total_datasets,
                "total_models": total_models,
                "total_cleaned_datasets": total_cleaned,
                "recent_datasets": recent_datasets.data,
                "recent_models": recent_models.data
            }
        
        result = safe_db_operation(
            operation,
            "dashboard",
            "Error al obtener estadísticas del dashboard"
        )
        
        # Si Supabase no está disponible, devolver datos mock
        if not result["success"]:
            return {
                "success": True,
                "message": "Modo local - Supabase no disponible",
                "data": {
                    "total_datasets": 0,
                    "total_models": 0,
                    "total_cleaned_datasets": 0,
                    "recent_datasets": [],
                    "recent_models": []
                }
            }
        
        return result
    
    # ============= UTILIDADES =============
    
    def check_connection(self) -> Dict[str, Any]:
        """Verifica la conexión a Supabase"""
        if self.client is None:
            return {
                "connected": False,
                "message": "Supabase no configurado"
            }
        
        def operation(client):
            # Intenta una query simple
            result = client.table("datasets").select("id").limit(1).execute()
            return {"test": "ok"}
        
        result = safe_db_operation(
            operation,
            "connection_test",
            "Error al verificar conexión"
        )
        
        return {
            "connected": result["success"],
            "message": result["message"]
        }

