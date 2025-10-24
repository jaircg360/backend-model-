"""
Servicio de entrenamiento de modelos usando Scikit-learn y PyTorch
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional
import joblib
import time
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score
)

from app.models.schemas import ModelType
from app.config import settings

logger = logging.getLogger(__name__)

# ============= PYTORCH MODELS =============

class SimpleNeuralNetwork(nn.Module):
    """Red neuronal simple para regresión o clasificación"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout: float = 0.2):
        super(SimpleNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TabularDataset(Dataset):
    """Dataset personalizado para datos tabulares"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y.ndim == 1 else torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============= TRAINER SERVICE =============

class ModelTrainerService:
    """Servicio para entrenar modelos de ML"""
    
    def __init__(self):
        self.model = None
        self.model_type: ModelType = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}
        self.training_time: float = 0.0
    
    def load_data(self, file_path: str, target_column: str, feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Carga datos desde archivo con preprocesamiento automático"""
        try:
            # Cargar con detección automática de delimitador
            if file_path.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path, sep=None, engine='python')
                except:
                    df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Formato de archivo no soportado")
            
            logger.info(f"📂 Archivo cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Validar que exista la columna objetivo
            if target_column not in df.columns:
                raise ValueError(f"Columna objetivo '{target_column}' no encontrada en el dataset")
            
            # Seleccionar features
            if feature_columns is None or len(feature_columns) == 0:
                feature_columns = [col for col in df.columns if col != target_column]
            
            # Validar features
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"Columnas no encontradas: {missing_features}")
            
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # PREPROCESAMIENTO: Codificar variables categóricas
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_columns) > 0:
                logger.info(f"🔄 Codificando {len(categorical_columns)} columnas categóricas: {list(categorical_columns)}")
                from sklearn.preprocessing import LabelEncoder
                
                for col in categorical_columns:
                    le = LabelEncoder()
                    # Convertir a string primero para manejar valores mixtos
                    X[col] = X[col].astype(str)
                    X[col] = le.fit_transform(X[col])
            
            # Codificar target si es categórico
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                logger.info(f"🎯 Codificando columna objetivo '{target_column}'")
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            else:
                # Intentar convertir a numérico
                y = pd.to_numeric(y, errors='coerce')
            
            # Convertir features a numérico (después de codificar)
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            # Eliminar valores nulos
            initial_rows = len(X)
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            removed_rows = initial_rows - len(X)
            if removed_rows > 0:
                logger.warning(f"⚠️  Eliminadas {removed_rows} filas con valores nulos")
            
            if len(X) == 0:
                raise ValueError("Dataset vacío después del preprocesamiento. Verifica que los datos sean válidos.")
            
            self.feature_names = feature_columns
            self.target_name = target_column
            
            logger.info(f"✅ Datos listos: {X.shape[0]} muestras, {X.shape[1]} features")
            
            return X, y
        
        except Exception as e:
            logger.error(f"❌ Error al cargar datos: {str(e)}")
            raise
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = None) -> None:
        """Divide los datos en entrenamiento y prueba"""
        if random_state is None:
            random_state = settings.RANDOM_STATE
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"📊 Datos divididos: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def train_sklearn_model(self, algorithm: str, hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Entrena un modelo de Scikit-learn"""
        if hyperparameters is None:
            hyperparameters = {}
        
        start_time = time.time()
        
        # Seleccionar modelo
        models_map = {
            # Regresión
            "linear_regression": LinearRegression,
            "random_forest_regressor": RandomForestRegressor,
            "gradient_boosting_regressor": GradientBoostingRegressor,
            "svr": SVR,
            "knn_regressor": KNeighborsRegressor,
            
            # Clasificación
            "logistic_regression": LogisticRegression,
            "random_forest_classifier": RandomForestClassifier,
            "gradient_boosting_classifier": GradientBoostingClassifier,
            "svc": SVC,
            "knn_classifier": KNeighborsClassifier,
            
            # Clustering
            "kmeans": KMeans,
            "dbscan": DBSCAN,
        }
        
        if algorithm not in models_map:
            raise ValueError(f"Algoritmo '{algorithm}' no soportado")
        
        # Crear y entrenar modelo
        model_class = models_map[algorithm]
        self.model = model_class(**hyperparameters)
        
        logger.info(f"🚀 Entrenando {algorithm}...")
        
        # Entrenar según tipo
        if algorithm in ["kmeans", "dbscan"]:
            # Clustering no supervisado
            self.model.fit(self.X_train)
            predictions = self.model.predict(self.X_train)
            
            self.metrics = {
                "silhouette_score": float(silhouette_score(self.X_train, predictions)),
                "n_clusters": int(len(np.unique(predictions))),
                "samples": int(len(self.X_train))
            }
        else:
            # Supervisado (regresión o clasificación)
            self.model.fit(self.X_train, self.y_train)
            predictions = self.model.predict(self.X_test)
            
            # Calcular métricas
            if self.model_type == ModelType.REGRESSION:
                self.metrics = {
                    "mse": float(mean_squared_error(self.y_test, predictions)),
                    "rmse": float(np.sqrt(mean_squared_error(self.y_test, predictions))),
                    "mae": float(mean_absolute_error(self.y_test, predictions)),
                    "r2_score": float(r2_score(self.y_test, predictions))
                }
            else:  # Clasificación
                # Manejo de clases múltiples
                average_method = 'binary' if len(np.unique(self.y_train)) == 2 else 'weighted'
                
                self.metrics = {
                    "accuracy": float(accuracy_score(self.y_test, predictions)),
                    "precision": float(precision_score(self.y_test, predictions, average=average_method, zero_division=0)),
                    "recall": float(recall_score(self.y_test, predictions, average=average_method, zero_division=0)),
                    "f1_score": float(f1_score(self.y_test, predictions, average=average_method, zero_division=0))
                }
            
            # Feature importance (si está disponible)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = {
                    name: float(importance)
                    for name, importance in zip(self.feature_names, self.model.feature_importances_)
                }
                # Ordenar por importancia
                self.feature_importance = dict(sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            elif hasattr(self.model, 'coef_'):
                # Para modelos lineales
                coef = self.model.coef_
                if coef.ndim > 1:
                    coef = np.abs(coef).mean(axis=0)
                self.feature_importance = {
                    name: float(abs(coef_val))
                    for name, coef_val in zip(self.feature_names, coef)
                }
        
        self.training_time = time.time() - start_time
        
        logger.info(f"✅ Modelo entrenado en {self.training_time:.2f}s")
        logger.info(f"📊 Métricas: {self.metrics}")
        
        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
            "training_time": self.training_time
        }
    
    def train_pytorch_model(self, hidden_sizes: List[int] = None, epochs: int = 100, 
                           batch_size: int = 32, learning_rate: float = 0.001,
                           task_type: str = "regression") -> Dict[str, Any]:
        """Entrena una red neuronal con PyTorch"""
        if hidden_sizes is None:
            hidden_sizes = [64, 32]
        
        start_time = time.time()
        
        # Preparar datos
        X_train_np = self.X_train.values
        X_test_np = self.X_test.values
        y_train_np = self.y_train.values.reshape(-1, 1) if task_type == "regression" else self.y_train.values
        y_test_np = self.y_test.values.reshape(-1, 1) if task_type == "regression" else self.y_test.values
        
        # Crear datasets
        train_dataset = TabularDataset(X_train_np, y_train_np)
        test_dataset = TabularDataset(X_test_np, y_test_np)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Crear modelo
        input_size = X_train_np.shape[1]
        output_size = 1 if task_type == "regression" else len(np.unique(y_train_np))
        
        self.model = SimpleNeuralNetwork(input_size, hidden_sizes, output_size)
        
        # Loss y optimizer
        if task_type == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"🚀 Entrenando red neuronal ({epochs} épocas)...")
        
        # Entrenamiento
        train_losses = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if task_type == "classification":
                    batch_y = batch_y.long()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Época {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluación
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                
                if task_type == "regression":
                    predictions = outputs.numpy()
                else:
                    predictions = torch.argmax(outputs, dim=1).numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(batch_y.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calcular métricas
        if task_type == "regression":
            self.metrics = {
                "mse": float(mean_squared_error(all_targets, all_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(all_targets, all_predictions))),
                "mae": float(mean_absolute_error(all_targets, all_predictions)),
                "r2_score": float(r2_score(all_targets, all_predictions)),
                "final_train_loss": float(train_losses[-1])
            }
        else:
            average_method = 'binary' if output_size == 2 else 'weighted'
            self.metrics = {
                "accuracy": float(accuracy_score(all_targets, all_predictions)),
                "precision": float(precision_score(all_targets, all_predictions, average=average_method, zero_division=0)),
                "recall": float(recall_score(all_targets, all_predictions, average=average_method, zero_division=0)),
                "f1_score": float(f1_score(all_targets, all_predictions, average=average_method, zero_division=0)),
                "final_train_loss": float(train_losses[-1])
            }
        
        self.training_time = time.time() - start_time
        
        logger.info(f"✅ Red neuronal entrenada en {self.training_time:.2f}s")
        logger.info(f"📊 Métricas: {self.metrics}")
        
        return {
            "metrics": self.metrics,
            "training_time": self.training_time,
            "train_losses": train_losses
        }
    
    def save_model(self, model_path: str, is_pytorch: bool = False) -> str:
        """Guarda el modelo entrenado"""
        try:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            
            if is_pytorch:
                torch.save(self.model.state_dict(), model_path)
            else:
                joblib.dump(self.model, model_path)
            
            logger.info(f"💾 Modelo guardado en: {model_path}")
            return model_path
        
        except Exception as e:
            logger.error(f"❌ Error al guardar modelo: {str(e)}")
            raise
    
    def load_model(self, model_path: str, is_pytorch: bool = False, model_config: Dict = None) -> Any:
        """Carga un modelo previamente entrenado"""
        try:
            if is_pytorch:
                if model_config is None:
                    raise ValueError("Se requiere configuración del modelo para PyTorch")
                
                self.model = SimpleNeuralNetwork(
                    model_config['input_size'],
                    model_config['hidden_sizes'],
                    model_config['output_size']
                )
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            else:
                self.model = joblib.load(model_path)
            
            logger.info(f"✅ Modelo cargado desde: {model_path}")
            return self.model
        
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Realiza predicciones con el modelo entrenado"""
        if self.model is None:
            raise ValueError("No hay modelo entrenado")
        
        if isinstance(self.model, nn.Module):
            # PyTorch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X.values)
                predictions = self.model(X_tensor).numpy()
        else:
            # Scikit-learn
            predictions = self.model.predict(X)
        
        return predictions
    
    def get_predictions_sample(self, n: int = 10) -> List[Dict[str, Any]]:
        """Obtiene una muestra de predicciones vs valores reales"""
        if self.X_test is None or self.y_test is None:
            return []
        
        predictions = self.predict(self.X_test.head(n))
        
        sample = []
        for i in range(min(n, len(predictions))):
            sample.append({
                "actual": float(self.y_test.iloc[i]),
                "predicted": float(predictions[i]) if predictions[i].ndim == 0 else float(predictions[i][0]),
                "features": self.X_test.iloc[i].to_dict()
            })
        
        return sample

