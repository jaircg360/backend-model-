"""
Endpoints para entrenamiento de modelos
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
import logging
from datetime import datetime

from app.services.model_trainer import ModelTrainerService
from app.services.supabase_service import SupabaseService
from app.services.agent_service import AgentService
from app.models.schemas import TrainingRequest, TrainingResponse, ModelType
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entrena un modelo de Machine Learning
    """
    try:
        # Buscar archivo
        supabase_service = SupabaseService()
        dataset_result = supabase_service.get_dataset(request.file_id)
        
        file_path = None
        if dataset_result["success"] and dataset_result["data"]:
            file_path = dataset_result["data"]["file_path"]
        else:
            uploads_dir = Path(settings.UPLOADS_DIR)
            matching_files = list(uploads_dir.glob(f"{request.file_id}*"))
            if matching_files:
                file_path = str(matching_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        model_type_str = request.model_type if isinstance(request.model_type, str) else request.model_type.value
        logger.info(f"🚀 Entrenando modelo {model_type_str} para {request.file_id}...")
        
        # Inicializar trainer
        trainer = ModelTrainerService()
        trainer.model_type = request.model_type
        
        # Cargar y dividir datos
        X, y = trainer.load_data(
            file_path=file_path,
            target_column=request.target_column,
            feature_columns=request.feature_columns
        )
        
        trainer.split_data(X, y, test_size=request.test_size)
        
        # Determinar algoritmo
        algorithm = request.algorithm
        if algorithm is None:
            # Seleccionar algoritmo por defecto
            if model_type_str == "regression":
                algorithm = "random_forest_regressor"
            elif model_type_str == "classification":
                algorithm = "random_forest_classifier"
            elif model_type_str == "clustering":
                algorithm = "kmeans"
            else:
                algorithm = "neural_network"
        
        # Entrenar modelo
        if algorithm == "neural_network" or model_type_str == "neural_network":
            # PyTorch
            task_type = "regression" if model_type_str == "regression" else "classification"
            hyperparams = request.hyperparameters or {}
            
            training_result = trainer.train_pytorch_model(
                hidden_sizes=hyperparams.get("hidden_sizes", [64, 32]),
                epochs=hyperparams.get("epochs", 100),
                batch_size=hyperparams.get("batch_size", 32),
                learning_rate=hyperparams.get("learning_rate", 0.001),
                task_type=task_type
            )
            is_pytorch = True
        else:
            # Scikit-learn
            training_result = trainer.train_sklearn_model(
                algorithm=algorithm,
                hyperparameters=request.hyperparameters or {}
            )
            is_pytorch = False
        
        # Guardar modelo
        model_id = f"model_{request.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_extension = ".pth" if is_pytorch else ".joblib"
        model_path = os.path.join(settings.MODELS_DIR, f"{model_id}{model_extension}")
        
        trainer.save_model(model_path, is_pytorch=is_pytorch)
        
        # Obtener muestra de predicciones
        predictions_sample = trainer.get_predictions_sample(n=5)
        
        # Guardar en Supabase
        model_data = {
            "model_name": f"{model_type_str}_{algorithm}",
            "model_type": model_type_str,
            "dataset_id": request.file_id,
            "model_path": model_path,
            "metrics": training_result["metrics"],
            "feature_importance": training_result.get("feature_importance"),
            "hyperparameters": request.hyperparameters or {},
            "status": "trained"
        }
        
        model_db_result = supabase_service.create_model(model_data)
        
        if model_db_result["success"] and model_db_result["data"]:
            model_id = model_db_result["data"]["id"]
        
        # Actualizar estado del dataset
        supabase_service.update_dataset(request.file_id, {"status": "trained"})
        
        logger.info(f"✅ Modelo entrenado exitosamente: {model_id}")
        
        return TrainingResponse(
            success=True,
            message=f"Modelo {model_type_str} entrenado exitosamente",
            model_id=model_id,
            model_type=model_type_str,
            metrics=training_result["metrics"],
            feature_importance=training_result.get("feature_importance"),
            training_time=training_result["training_time"],
            predictions_sample=predictions_sample
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al entrenar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al entrenar modelo: {str(e)}")

@router.get("/suggestions/{file_id}")
async def get_training_suggestions(file_id: str, target_column: str):
    """
    Obtiene sugerencias de modelos y algoritmos para un dataset
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
            matching_files = list(uploads_dir.glob(f"{file_id}*"))
            if matching_files:
                file_path = str(matching_files[0])
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        # Analizar con el agente
        agent = AgentService()
        suggestions = agent.suggest_model_type(
            target_column=target_column,
            file_path=file_path
        )
        
        if not suggestions["success"]:
            raise HTTPException(status_code=500, detail=suggestions.get("message", "Error al analizar"))
        
        return {
            "success": True,
            "file_id": file_id,
            "suggestions": suggestions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al obtener sugerencias: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{file_id}")
async def get_models_by_dataset(file_id: str):
    """
    Obtiene todos los modelos entrenados para un dataset
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_models_by_dataset(file_id)
        
        if result["success"]:
            return {
                "success": True,
                "file_id": file_id,
                "models": result["data"] or []
            }
        else:
            return {
                "success": True,
                "file_id": file_id,
                "models": [],
                "message": result["message"]
            }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/{model_id}")
async def get_model_info(model_id: str):
    """
    Obtiene información detallada de un modelo
    """
    try:
        supabase_service = SupabaseService()
        result = supabase_service.get_model(model_id)
        
        if result["success"] and result["data"]:
            return {
                "success": True,
                "model": result["data"]
            }
        else:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al obtener modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """
    Elimina un modelo
    """
    try:
        supabase_service = SupabaseService()
        
        # Obtener info del modelo
        model_result = supabase_service.get_model(model_id)
        
        if model_result["success"] and model_result["data"]:
            model_path = model_result["data"]["model_path"]
            
            # Eliminar archivo del modelo
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"🗑️ Archivo de modelo eliminado: {model_path}")
            
            # Eliminar de BD
            supabase_service.delete_model(model_id)
        
        return {
            "success": True,
            "message": "Modelo eliminado exitosamente"
        }
    
    except Exception as e:
        logger.error(f"❌ Error al eliminar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

