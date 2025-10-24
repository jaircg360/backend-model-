"""
FastAPI Backend Principal para Model Prep Pro
Sistema de limpieza, entrenamiento y modelado de datos
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging

from app.config import settings
from app.routes import upload, clean, train, agent, dashboard, charts

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear directorios necesarios
for directory in ["uploads", "models", "exports", "charts"]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"📁 Directorio verificado: {directory}")

app = FastAPI(
    title="Model Prep Pro API",
    description="Backend para procesamiento, limpieza y entrenamiento de modelos de ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ✅ CONFIGURAR CORS PARA VERCEL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-model-zeta.vercel.app",  # ⭐ Tu frontend en producción
        "http://localhost:5173",                    # Desarrollo local (Vite)
        "http://localhost:3000",                    # Desarrollo local (React/Next)
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"  # ⚠️ Temporalmente permite todo (eliminar en producción final)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"❌ Error global: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Error interno del servidor",
            "error": str(exc)
        }
    )

# Rutas principales
@app.get("/")
async def root():
    return {
        "message": "Model Prep Pro API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "frontend": "https://frontend-model-zeta.vercel.app",
        "environment": "production" if not settings.DEBUG else "development"
    }

@app.get("/health")
async def health_check():
    """Health check para monitoreo"""
    return {
        "status": "healthy",
        "cors_enabled": True,
        "frontend_url": "https://frontend-model-zeta.vercel.app",
        "supabase": settings.check_supabase_connection()
    }

@app.get("/ping")
async def ping():
    """Keep-alive endpoint para evitar hibernación"""
    return {"status": "pong", "message": "Backend activo"}

# Incluir routers
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(clean.router, prefix="/api/clean", tags=["Clean"])
app.include_router(train.router, prefix="/api/train", tags=["Train"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(charts.router, prefix="/api", tags=["Charts"])

logger.info("✅ Model Prep Pro API iniciada correctamente")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # ⚠️ Desactivado para producción
    )
