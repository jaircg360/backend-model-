"""
Configuración del backend
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Configuración general
    APP_NAME: str = "Model Prep Pro"
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    LOG_LEVEL: str = "INFO"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Límites de archivos
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set = {".csv", ".xlsx", ".xls"}
    
    # Configuración de modelos
    MODELS_DIR: str = "models"
    UPLOADS_DIR: str = "uploads"
    EXPORTS_DIR: str = "exports"
    
    # Machine Learning
    DEFAULT_TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    def check_supabase_connection(self) -> dict:
        """Verifica si la conexión a Supabase está configurada"""
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            return {
                "connected": False,
                "message": "Supabase no configurado. Por favor, configura SUPABASE_URL y SUPABASE_KEY en el archivo .env"
            }
        
        try:
            from app.database import supabase_client
            # Intentar una operación simple
            if supabase_client:
                return {
                    "connected": True,
                    "message": "Conexión a Supabase exitosa"
                }
        except Exception as e:
            return {
                "connected": False,
                "message": f"Error al conectar con Supabase: {str(e)}"
            }
        
        return {
            "connected": False,
            "message": "No se pudo verificar la conexión"
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignorar campos extra del .env

settings = Settings()

