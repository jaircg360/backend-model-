"""
Validadores y utilidades
"""

import os
from pathlib import Path
from typing import Optional
import magic
import logging

logger = logging.getLogger(__name__)

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Valida que la extensión del archivo sea permitida"""
    extension = Path(filename).suffix.lower()
    return extension in allowed_extensions

def validate_file_size(file_path: str, max_size_bytes: int) -> bool:
    """Valida que el tamaño del archivo no exceda el límite"""
    if not os.path.exists(file_path):
        return False
    
    file_size = os.path.getsize(file_path)
    return file_size <= max_size_bytes

def get_file_mime_type(file_path: str) -> Optional[str]:
    """Obtiene el tipo MIME de un archivo"""
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        logger.warning(f"No se pudo determinar el tipo MIME: {str(e)}")
        return None

def sanitize_filename(filename: str) -> str:
    """Limpia un nombre de archivo de caracteres peligrosos"""
    # Eliminar caracteres no seguros
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Limitar longitud
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized

def validate_column_name(column_name: str, valid_columns: list) -> bool:
    """Valida que un nombre de columna exista en la lista válida"""
    return column_name in valid_columns

def format_bytes(bytes_size: int) -> str:
    """Formatea bytes a formato legible (KB, MB, GB)"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

