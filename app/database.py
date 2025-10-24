"""
Configuración de base de datos Supabase
"""

from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Cliente de Supabase
supabase_client = None

def get_supabase_client():
    """Obtiene el cliente de Supabase"""
    global supabase_client
    
    if supabase_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("⚠️  Supabase no configurado. Funcionando en modo local.")
            return None
        
        try:
            # Intentar importar supabase solo si está configurado
            from supabase import create_client
            supabase_client = create_client(
                supabase_url=settings.SUPABASE_URL,
                supabase_key=settings.SUPABASE_KEY
            )
            logger.info("✅ Conexión a Supabase establecida")
        except ImportError:
            logger.warning("⚠️  Librería supabase no instalada. Funcionando en modo local.")
            return None
        except Exception as e:
            logger.error(f"❌ Error al conectar con Supabase: {str(e)}")
            return None
    
    return supabase_client

def safe_db_operation(operation, table_name: str, error_message: str = "Error en operación de BD"):
    """
    Wrapper para operaciones de BD con manejo de errores
    """
    try:
        client = get_supabase_client()
        if client is None:
            return {
                "success": False,
                "message": "Base de datos no disponible. Trabajando en modo local.",
                "data": None
            }
        
        result = operation(client)
        return {
            "success": True,
            "message": "Operación exitosa",
            "data": result
        }
    
    except Exception as e:
        logger.error(f"❌ Error en {table_name}: {str(e)}")
        return {
            "success": False,
            "message": f"{error_message}: {str(e)}",
            "data": None
        }

# Inicializar cliente al importar
get_supabase_client()

