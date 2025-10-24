"""
Servicio de limpieza de datos usando pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

from app.models.schemas import CleaningAction

logger = logging.getLogger(__name__)

class DataCleanerService:
    """Servicio para limpiar y preprocesar datos"""
    
    def __init__(self):
        self.df: pd.DataFrame = None
        self.original_shape: Tuple[int, int] = None
        self.applied_actions: List[str] = []
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carga datos desde un archivo CSV o Excel con detección automática de delimitador"""
        try:
            if file_path.endswith('.csv'):
                # Intentar detectar automáticamente el delimitador
                # Primero intentamos con sep=None para auto-detección
                try:
                    self.df = pd.read_csv(file_path, sep=None, engine='python')
                    logger.info(f"🔍 Delimitador detectado automáticamente")
                except Exception:
                    # Si falla, intentar delimitadores comunes
                    for sep in [',', ';', '\t', '|']:
                        try:
                            self.df = pd.read_csv(file_path, sep=sep)
                            if self.df.shape[1] > 1:  # Si tiene más de 1 columna, es correcto
                                logger.info(f"🔍 Delimitador detectado: '{sep}'")
                                break
                        except Exception:
                            continue
                    
                    # Si ninguno funciona, usar el predeterminado
                    if self.df is None or self.df.shape[1] == 1:
                        self.df = pd.read_csv(file_path)
                        logger.warning("⚠️  Usando delimitador predeterminado (coma)")
                        
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Formato de archivo no soportado")
            
            self.original_shape = self.df.shape
            logger.info(f"✅ Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
            return self.df
        
        except Exception as e:
            logger.error(f"❌ Error al cargar datos: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen completo de los datos"""
        if self.df is None:
            return {}
        
        summary = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "duplicates": int(self.df.duplicated().sum()),
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Estadísticas para columnas numéricas
        if len(summary["numeric_columns"]) > 0:
            summary["numeric_stats"] = self.df[summary["numeric_columns"]].describe().to_dict()
        
        return summary
    
    def remove_duplicates(self) -> int:
        """Elimina filas duplicadas"""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            self.applied_actions.append(f"Eliminadas {removed} filas duplicadas")
            logger.info(f"🧹 Eliminadas {removed} filas duplicadas")
        
        return removed
    
    def handle_missing_values(self, strategy: str = "mean", columns: List[str] = None) -> Dict[str, int]:
        """
        Maneja valores faltantes
        
        Args:
            strategy: 'mean', 'median', 'mode', 'drop', 'fill_value'
            columns: Columnas específicas a procesar (None = todas)
        """
        if columns is None:
            columns = self.df.columns.tolist()
        
        filled_counts = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == "mean" and self.df[col].dtype in [np.float64, np.int64]:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
                filled_counts[col] = int(missing_count)
                self.applied_actions.append(f"{col}: rellenado con media ({missing_count} valores)")
            
            elif strategy == "median" and self.df[col].dtype in [np.float64, np.int64]:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                filled_counts[col] = int(missing_count)
                self.applied_actions.append(f"{col}: rellenado con mediana ({missing_count} valores)")
            
            elif strategy == "mode":
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                if mode_value is not None:
                    self.df[col].fillna(mode_value, inplace=True)
                    filled_counts[col] = int(missing_count)
                    self.applied_actions.append(f"{col}: rellenado con moda ({missing_count} valores)")
            
            elif strategy == "drop":
                self.df = self.df.dropna(subset=[col])
                filled_counts[col] = int(missing_count)
                self.applied_actions.append(f"{col}: eliminadas {missing_count} filas con valores nulos")
        
        if filled_counts:
            logger.info(f"🧹 Valores faltantes procesados: {filled_counts}")
        
        return filled_counts
    
    def normalize_columns(self, columns: List[str] = None, method: str = "minmax") -> List[str]:
        """
        Normaliza o estandariza columnas numéricas
        
        Args:
            columns: Columnas a normalizar (None = todas las numéricas)
            method: 'minmax' (0-1) o 'standard' (z-score)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized = []
        
        for col in columns:
            if col not in self.df.columns or self.df[col].dtype not in [np.float64, np.int64]:
                continue
            
            if method == "minmax":
                scaler = MinMaxScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                normalized.append(col)
                self.applied_actions.append(f"{col}: normalizado (MinMax 0-1)")
            
            elif method == "standard":
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                normalized.append(col)
                self.applied_actions.append(f"{col}: estandarizado (Z-score)")
        
        if normalized:
            logger.info(f"📊 Normalizadas {len(normalized)} columnas con método {method}")
        
        return normalized
    
    def encode_categorical(self, columns: List[str] = None, method: str = "label") -> Dict[str, Any]:
        """
        Codifica variables categóricas
        
        Args:
            columns: Columnas a codificar (None = todas las categóricas)
            method: 'label' o 'onehot'
        """
        if columns is None:
            columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        encoded_info = {}
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == "label":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                encoded_info[col] = {
                    "method": "label",
                    "classes": le.classes_.tolist()
                }
                self.applied_actions.append(f"{col}: codificado (Label Encoding)")
            
            elif method == "onehot":
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df = self.df.drop(columns=[col])
                encoded_info[col] = {
                    "method": "onehot",
                    "new_columns": dummies.columns.tolist()
                }
                self.applied_actions.append(f"{col}: codificado (One-Hot Encoding)")
        
        if encoded_info:
            logger.info(f"🔤 Codificadas {len(encoded_info)} columnas categóricas")
        
        return encoded_info
    
    def remove_outliers(self, columns: List[str] = None, method: str = "iqr", threshold: float = 1.5) -> int:
        """
        Elimina outliers usando IQR o Z-score
        
        Args:
            columns: Columnas a procesar (None = todas las numéricas)
            method: 'iqr' o 'zscore'
            threshold: 1.5 para IQR, 3 para Z-score
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            elif method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        
        removed = initial_rows - len(self.df)
        
        if removed > 0:
            self.applied_actions.append(f"Eliminadas {removed} filas con outliers (método: {method})")
            logger.info(f"🎯 Eliminados {removed} outliers usando {method}")
        
        return removed
    
    def apply_cleaning_pipeline(self, actions: List[CleaningAction], **kwargs) -> Dict[str, Any]:
        """
        Aplica un pipeline de limpieza completo
        
        Args:
            actions: Lista de acciones a aplicar
            **kwargs: Parámetros adicionales para cada acción
        """
        self.applied_actions = []
        results = {
            "original_shape": self.original_shape,
            "actions_results": {}
        }
        
        for action in actions:
            try:
                if action == CleaningAction.REMOVE_DUPLICATES:
                    removed = self.remove_duplicates()
                    results["actions_results"]["duplicates_removed"] = removed
                
                elif action == CleaningAction.FILL_NULLS_MEAN:
                    filled = self.handle_missing_values(strategy="mean")
                    results["actions_results"]["nulls_filled_mean"] = filled
                
                elif action == CleaningAction.FILL_NULLS_MEDIAN:
                    filled = self.handle_missing_values(strategy="median")
                    results["actions_results"]["nulls_filled_median"] = filled
                
                elif action == CleaningAction.FILL_NULLS_MODE:
                    filled = self.handle_missing_values(strategy="mode")
                    results["actions_results"]["nulls_filled_mode"] = filled
                
                elif action == CleaningAction.DROP_NULLS:
                    filled = self.handle_missing_values(strategy="drop")
                    results["actions_results"]["nulls_dropped"] = filled
                
                elif action == CleaningAction.NORMALIZE:
                    normalized = self.normalize_columns(method="minmax")
                    results["actions_results"]["normalized_columns"] = normalized
                
                elif action == CleaningAction.STANDARDIZE:
                    standardized = self.normalize_columns(method="standard")
                    results["actions_results"]["standardized_columns"] = standardized
                
                elif action == CleaningAction.ENCODE_CATEGORICAL:
                    encoding_method = kwargs.get("encoding_method", "label")
                    encoded = self.encode_categorical(method=encoding_method)
                    results["actions_results"]["encoded_columns"] = encoded
                
                elif action == CleaningAction.REMOVE_OUTLIERS:
                    outlier_method = kwargs.get("outlier_method", "iqr")
                    removed = self.remove_outliers(method=outlier_method)
                    results["actions_results"]["outliers_removed"] = removed
            
            except Exception as e:
                logger.error(f"❌ Error en acción {action}: {str(e)}")
                results["actions_results"][f"error_{action}"] = str(e)
        
        results["final_shape"] = self.df.shape
        results["applied_actions"] = self.applied_actions
        results["summary"] = self.get_data_summary()
        
        return results
    
    def save_cleaned_data(self, output_path: str) -> str:
        """Guarda los datos limpios"""
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"💾 Datos limpios guardados en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Error al guardar datos: {str(e)}")
            raise
    
    def get_preview(self, n: int = 10) -> List[Dict[str, Any]]:
        """Obtiene una vista previa de los datos"""
        if self.df is None:
            return []
        
        return self.df.head(n).fillna("NULL").to_dict(orient="records")

