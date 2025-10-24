"""
Endpoints para el Agente Inteligente
"""

from fastapi import APIRouter, HTTPException
import logging

from app.services.agent_service import AgentService
from app.models.schemas import AgentRequest, AgentResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Instancia global del agente (mantiene el historial de conversación)
agent = AgentService()

@router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """
    Chatea con el agente inteligente
    """
    try:
        logger.info(f"💬 Usuario: {request.message}")
        
        response = agent.process_user_message(
            message=request.message,
            context=request.context
        )
        
        logger.info(f"🤖 Agente: {response['message'][:100]}...")
        
        return AgentResponse(
            success=response["success"],
            message=response["message"],
            suggestions=response.get("suggestions"),
            next_step=response.get("next_step"),
            warning=None
        )
    
    except Exception as e:
        logger.error(f"❌ Error en chat con agente: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{file_id}")
async def analyze_dataset_with_agent(file_id: str):
    """
    Analiza un dataset y proporciona recomendaciones del agente
    """
    try:
        from pathlib import Path
        from app.config import settings
        from app.services.supabase_service import SupabaseService
        
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
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        analysis = agent.analyze_dataset(file_path)
        
        if not analysis["success"]:
            raise HTTPException(status_code=500, detail=analysis.get("message"))
        
        return {
            "success": True,
            "file_id": file_id,
            "analysis": analysis["analysis"],
            "suggestions": analysis["suggestions"],
            "next_step": analysis["next_step"],
            "warnings": analysis["warnings"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al analizar dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_conversation_history():
    """
    Obtiene el historial de conversación con el agente
    """
    try:
        history = agent.get_conversation_history()
        
        return {
            "success": True,
            "history": history,
            "total_messages": len(history)
        }
    
    except Exception as e:
        logger.error(f"❌ Error al obtener historial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history")
async def clear_conversation_history():
    """
    Limpia el historial de conversación
    """
    try:
        agent.clear_history()
        
        return {
            "success": True,
            "message": "Historial de conversación limpiado"
        }
    
    except Exception as e:
        logger.error(f"❌ Error al limpiar historial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/help")
async def get_help():
    """
    Obtiene información de ayuda del agente
    """
    return {
        "success": True,
        "message": "Agente Inteligente de Model Prep Pro",
        "capabilities": [
            "Analizar datasets y detectar problemas",
            "Recomendar acciones de limpieza",
            "Sugerir modelos de ML apropiados",
            "Guiar paso a paso en el proceso",
            "Responder preguntas sobre ML",
            "Explicar métricas y resultados"
        ],
        "example_questions": [
            "¿Cómo limpio mis datos?",
            "¿Qué modelo debo usar para mi dataset?",
            "Explícame las métricas de mi modelo",
            "¿Qué hago con los valores faltantes?",
            "¿Cuál es el siguiente paso?",
            "Ayuda con clasificación binaria"
        ]
    }

@router.post("/suggest-model/{file_id}")
async def suggest_model_for_dataset(file_id: str, target_column: str):
    """
    Sugiere el mejor tipo de modelo para un dataset y columna objetivo
    """
    try:
        from pathlib import Path
        from app.config import settings
        from app.services.supabase_service import SupabaseService
        
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
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        
        suggestions = agent.suggest_model_type(
            target_column=target_column,
            file_path=file_path
        )
        
        if not suggestions["success"]:
            raise HTTPException(status_code=500, detail=suggestions.get("message"))
        
        return suggestions
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error al sugerir modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

