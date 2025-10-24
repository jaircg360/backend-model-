#!/usr/bin/env python
"""
Script de inicio rápido para el backend de Model Prep Pro
"""

import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import torch
        logger.info("✅ Todas las dependencias principales están instaladas")
        
        # Verificar Supabase (opcional)
        try:
            import supabase
            logger.info("✅ Supabase disponible")
        except ImportError:
            logger.info("ℹ️  Supabase no instalado (opcional - modo local activo)")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Falta una dependencia: {str(e)}")
        logger.info("💡 Ejecuta: pip install -r requirements.txt")
        return False

def check_directories():
    """Crea los directorios necesarios si no existen"""
    directories = ['uploads', 'models', 'exports']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"📁 Directorio creado: {directory}")
        else:
            logger.info(f"✅ Directorio existente: {directory}")

def check_env_file():
    """Verifica que exista el archivo .env"""
    if not os.path.exists('.env'):
        logger.warning("⚠️  Archivo .env no encontrado")
        logger.info("💡 Copia env.example a .env y configura tus variables")
        logger.info("   El servidor funcionará en modo local sin Supabase")
        return False
    else:
        logger.info("✅ Archivo .env encontrado")
        return True

def main():
    """Función principal"""
    logger.info("🚀 Iniciando Model Prep Pro Backend...")
    logger.info("=" * 60)
    
    # Verificaciones
    logger.info("🔍 Verificando sistema...")
    
    if not check_dependencies():
        sys.exit(1)
    
    check_directories()
    check_env_file()
    
    logger.info("=" * 60)
    logger.info("✅ Sistema listo para iniciar")
    logger.info("🌐 Servidor: http://localhost:8000")
    logger.info("📚 Documentación: http://localhost:8000/docs")
    logger.info("=" * 60)
    logger.info("")
    
    # Importar y ejecutar
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error al iniciar servidor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

