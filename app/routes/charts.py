"""
Endpoints para gráficos y visualizaciones de modelos
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime

from ..services.chart_service import ChartService
from ..services.supabase_service import SupabaseService
from ..models.schemas import ModelInfo

router = APIRouter(prefix="/charts", tags=["charts"])

chart_service = ChartService()
supabase_service = SupabaseService()

@router.get("/model/{model_id}")
async def get_model_charts(model_id: str) -> Dict[str, Any]:
    """Obtiene todos los gráficos para un modelo específico"""
    try:
        # Buscar información del modelo
        model_info = await get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        # Generar gráficos
        charts = chart_service.generate_model_charts(model_id, model_info)
        
        # Convertir rutas a base64 para el frontend
        charts_base64 = {}
        for chart_name, chart_path in charts.items():
            if chart_path and os.path.exists(chart_path):
                charts_base64[chart_name] = chart_service.get_chart_base64(chart_path)
        
        return {
            "success": True,
            "model_id": model_id,
            "model_info": model_info,
            "charts": charts_base64,
            "chart_count": len(charts_base64)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando gráficos: {str(e)}")

@router.get("/model/{model_id}/chart/{chart_type}")
async def get_specific_chart(model_id: str, chart_type: str) -> Dict[str, Any]:
    """Obtiene un gráfico específico de un modelo"""
    try:
        # Buscar información del modelo
        model_info = await get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        # Generar solo el gráfico específico
        charts = chart_service.generate_model_charts(model_id, model_info)
        
        if chart_type not in charts:
            raise HTTPException(status_code=404, detail=f"Gráfico {chart_type} no encontrado")
        
        chart_path = charts[chart_type]
        if not chart_path or not os.path.exists(chart_path):
            raise HTTPException(status_code=404, detail=f"Archivo de gráfico no encontrado")
        
        chart_base64 = chart_service.get_chart_base64(chart_path)
        
        return {
            "success": True,
            "model_id": model_id,
            "chart_type": chart_type,
            "chart_data": chart_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo gráfico: {str(e)}")

@router.get("/models/summary")
async def get_all_models_summary() -> Dict[str, Any]:
    """Obtiene un resumen de todos los modelos con sus gráficos disponibles"""
    try:
        models_summary = []
        
        # Obtener lista de modelos desde Supabase o archivos locales
        if supabase_service.is_available():
            models = await supabase_service.get_all_models()
        else:
            # Modo local - buscar archivos de modelos
            models_dir = "backend/models"
            models = []
            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    if filename.endswith('.joblib'):
                        model_id = filename.replace('.joblib', '')
                        # Cargar información del modelo desde el archivo
                        try:
                            model_path = os.path.join(models_dir, filename)
                            model_data = joblib.load(model_path)

                            # Determinar el tipo de modelo basado en la clase sklearn
                            model_class = model_data.get('model', None).__class__.__name__ if model_data.get('model', None) else ''
                            if 'Regressor' in model_class:
                                detected_model_type = 'regression'
                            elif 'Classifier' in model_class:
                                detected_model_type = 'classification'
                            elif 'KMeans' in model_class or 'Clustering' in model_class:
                                detected_model_type = 'clustering'
                            else:
                                detected_model_type = model_data.get('model_type', 'unknown')

                            # Extraer información del modelo
                            model_info = {
                                "model_id": model_id,
                                "model_type": detected_model_type,
                                "created_at": model_data.get('created_at', datetime.now().isoformat()),
                                "dataset_name": model_data.get('dataset_name', 'local_dataset'),
                                "metrics": model_data.get('metrics', {}),
                                "feature_columns": model_data.get('feature_columns', []),
                                "target_column": model_data.get('target_column', ''),
                                "training_time": model_data.get('training_time', 0),
                                "test_size": model_data.get('test_size', 0.2),
                                "random_state": model_data.get('random_state', 42),
                                "cv_folds": model_data.get('cv_folds', 5)
                            }
                            models.append(model_info)
                        except Exception as e:
                            print(f"Error cargando modelo {model_id}: {str(e)}")
                            # Agregar modelo con información básica si falla la carga
                            models.append({
                                "model_id": model_id,
                                "model_type": "unknown",
                                "created_at": datetime.now().isoformat(),
                                "dataset_name": "local_dataset"
                            })
        
        # Para cada modelo, obtener información básica
        for model in models:
            model_id = model.get('model_id', '')
            try:
                # En modo local, usar la información ya cargada del modelo
                if not supabase_service.is_available():
                    model_info = model  # Ya tenemos toda la info del modelo
                else:
                    model_info = await get_model_info(model_id)

                if model_info:
                    # Verificar qué gráficos están disponibles
                    charts = chart_service.generate_model_charts(model_id, model_info)
                    available_charts = [name for name, path in charts.items() if path and os.path.exists(path)]

                    models_summary.append({
                        "model_id": model_id,
                        "model_type": model_info.get('model_type', 'unknown'),
                        "dataset_name": model_info.get('dataset_name', 'unknown'),
                        "created_at": model_info.get('created_at', ''),
                        "metrics": model_info.get('metrics', {}),
                        "available_charts": available_charts,
                        "chart_count": len(available_charts)
                    })
            except Exception as e:
                print(f"Error procesando modelo {model_id}: {str(e)}")
                continue
        
        return {
            "success": True,
            "models": models_summary,
            "total_models": len(models_summary)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo resumen: {str(e)}")

@router.post("/model/{model_id}/regenerate")
async def regenerate_model_charts(model_id: str) -> Dict[str, Any]:
    """Regenera todos los gráficos para un modelo específico"""
    try:
        # Buscar información del modelo
        model_info = await get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        # Eliminar gráficos existentes
        charts_dir = "backend/charts"
        if os.path.exists(charts_dir):
            for filename in os.listdir(charts_dir):
                if filename.startswith(model_id):
                    os.remove(os.path.join(charts_dir, filename))
        
        # Regenerar gráficos
        charts = chart_service.generate_model_charts(model_id, model_info)
        
        # Convertir a base64
        charts_base64 = {}
        for chart_name, chart_path in charts.items():
            if chart_path and os.path.exists(chart_path):
                charts_base64[chart_name] = chart_service.get_chart_base64(chart_path)
        
        return {
            "success": True,
            "model_id": model_id,
            "message": f"Gráficos regenerados para modelo {model_id}",
            "charts": charts_base64,
            "chart_count": len(charts_base64)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerando gráficos: {str(e)}")

@router.get("/chart-types")
async def get_available_chart_types() -> Dict[str, Any]:
    """Obtiene los tipos de gráficos disponibles según el tipo de modelo"""
    try:
        chart_types = {
            "regression": [
                "prediction_vs_actual",
                "residuals", 
                "error_distribution",
                "feature_importance",
                "model_metrics",
                "training_summary"
            ],
            "classification": [
                "confusion_matrix",
                "roc_curve",
                "probability_distribution",
                "feature_importance",
                "model_metrics",
                "training_summary"
            ],
            "clustering": [
                "clusters_2d",
                "cluster_distribution",
                "feature_importance",
                "model_metrics",
                "training_summary"
            ],
            "common": [
                "feature_importance",
                "model_metrics",
                "training_summary"
            ]
        }
        
        return {
            "success": True,
            "chart_types": chart_types,
            "descriptions": {
                "prediction_vs_actual": "Predicciones vs Valores Reales",
                "residuals": "Análisis de Residuos",
                "error_distribution": "Distribución de Errores",
                "confusion_matrix": "Matriz de Confusión",
                "roc_curve": "Curva ROC",
                "probability_distribution": "Distribución de Probabilidades",
                "clusters_2d": "Visualización 2D de Clusters",
                "cluster_distribution": "Distribución de Clusters",
                "feature_importance": "Importancia de Características",
                "model_metrics": "Métricas del Modelo",
                "training_summary": "Resumen del Entrenamiento"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo tipos de gráficos: {str(e)}")

async def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Obtiene información de un modelo desde Supabase o archivos locales"""
    try:
        if supabase_service.is_available():
            # Buscar en Supabase
            model_data = await supabase_service.get_model_by_id(model_id)
            if model_data:
                return model_data
        else:
            # Modo local - cargar desde archivo
            model_path = f"backend/models/{model_id}.joblib"
            if os.path.exists(model_path):
                import joblib
                model_data = joblib.load(model_path)

                # Crear información completa del modelo
                return {
                    "model_id": model_id,
                    "model_type": model_data.get('model_type', 'local_model'),
                    "dataset_name": model_data.get('dataset_name', 'local_dataset'),
                    "created_at": model_data.get('created_at', datetime.now().isoformat()),
                    "metrics": model_data.get('metrics', {}),
                    "feature_columns": model_data.get('feature_columns', []),
                    "target_column": model_data.get('target_column', ''),
                    "training_time": model_data.get('training_time', 0),
                    "test_size": model_data.get('test_size', 0.2),
                    "random_state": model_data.get('random_state', 42),
                    "cv_folds": model_data.get('cv_folds', 5),
                    "y_test": model_data.get('y_test', []),
                    "y_pred": model_data.get('y_pred', []),
                    "y_pred_proba": model_data.get('y_pred_proba', []),
                    "X_test": model_data.get('X_test', []),
                    "cluster_labels": model_data.get('cluster_labels', []),
                    "feature_importance": model_data.get('feature_importance', {})
                }

        return None

    except Exception as e:
        print(f"Error obteniendo información del modelo {model_id}: {str(e)}")
        return None


