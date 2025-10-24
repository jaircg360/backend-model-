"""
FastAPI Backend Principal para Model Prep Pro
Sistema de limpieza, entrenamiento y modelado de datos
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os

from app.config import settings
from app.routes import upload, clean, train, agent, dashboard, charts

# Crear directorios necesarios
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("exports", exist_ok=True)
os.makedirs("charts", exist_ok=True)

app = FastAPI(
    title="Model Prep Pro API",
    description="Backend para procesamiento, limpieza y entrenamiento de modelos de ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manejador de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
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
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "supabase": settings.check_supabase_connection()
    }

# Incluir routers
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(clean.router, prefix="/api/clean", tags=["Clean"])
app.include_router(train.router, prefix="/api/train", tags=["Train"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(charts.router, prefix="/api", tags=["Charts"])

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

