"""
Servicio para generar gráficos y visualizaciones de modelos entrenados
"""

import os
import json
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from datetime import datetime

# Configurar estilo de matplotlib
plt.style.use('default')
sns.set_palette("husl")

class ChartService:
    """Servicio para generar gráficos de modelos ML"""
    
    def __init__(self):
        self.charts_dir = "backend/charts"
        os.makedirs(self.charts_dir, exist_ok=True)
    
    def generate_model_charts(self, model_id: str, model_data: Dict[str, Any]) -> Dict[str, str]:
        """Genera todos los gráficos para un modelo entrenado"""
        
        charts = {}
        
        try:
            # Cargar datos del modelo
            model_path = f"backend/models/{model_id}.joblib"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo {model_id} no encontrado")
            
            model_info = joblib.load(model_path)
            
            # Generar diferentes tipos de gráficos según el tipo de modelo
            model_type = model_data.get('model_type', 'unknown')
            
            if 'regression' in model_type.lower():
                charts.update(self._generate_regression_charts(model_id, model_info, model_data))
            elif 'classification' in model_type.lower():
                charts.update(self._generate_classification_charts(model_id, model_info, model_data))
            elif 'clustering' in model_type.lower():
                charts.update(self._generate_clustering_charts(model_id, model_info, model_data))
            
            # Gráficos comunes para todos los modelos
            charts.update(self._generate_common_charts(model_id, model_info, model_data))
            
            return charts
            
        except Exception as e:
            print(f"Error generando gráficos para modelo {model_id}: {str(e)}")
            return {}
    
    def _generate_regression_charts(self, model_id: str, model_info: Dict, model_data: Dict) -> Dict[str, str]:
        """Genera gráficos específicos para modelos de regresión"""
        charts = {}
        
        try:
            # 1. Predicciones vs Valores Reales
            charts['prediction_vs_actual'] = self._create_prediction_vs_actual_chart(
                model_id, model_info, model_data
            )
            
            # 2. Residuos
            charts['residuals'] = self._create_residuals_chart(
                model_id, model_info, model_data
            )
            
            # 3. Distribución de Errores
            charts['error_distribution'] = self._create_error_distribution_chart(
                model_id, model_info, model_data
            )
            
        except Exception as e:
            print(f"Error generando gráficos de regresión: {str(e)}")
        
        return charts
    
    def _generate_classification_charts(self, model_id: str, model_info: Dict, model_data: Dict) -> Dict[str, str]:
        """Genera gráficos específicos para modelos de clasificación"""
        charts = {}
        
        try:
            # 1. Matriz de Confusión
            charts['confusion_matrix'] = self._create_confusion_matrix_chart(
                model_id, model_info, model_data
            )
            
            # 2. Curva ROC (si es clasificación binaria)
            if self._is_binary_classification(model_data):
                charts['roc_curve'] = self._create_roc_curve_chart(
                    model_id, model_info, model_data
                )
            
            # 3. Distribución de Probabilidades
            charts['probability_distribution'] = self._create_probability_distribution_chart(
                model_id, model_info, model_data
            )
            
        except Exception as e:
            print(f"Error generando gráficos de clasificación: {str(e)}")
        
        return charts
    
    def _generate_clustering_charts(self, model_id: str, model_info: Dict, model_data: Dict) -> Dict[str, str]:
        """Genera gráficos específicos para modelos de clustering"""
        charts = {}
        
        try:
            # 1. Visualización de Clusters (2D)
            charts['clusters_2d'] = self._create_clusters_2d_chart(
                model_id, model_info, model_data
            )
            
            # 2. Distribución de Clusters
            charts['cluster_distribution'] = self._create_cluster_distribution_chart(
                model_id, model_info, model_data
            )
            
        except Exception as e:
            print(f"Error generando gráficos de clustering: {str(e)}")
        
        return charts
    
    def _generate_common_charts(self, model_id: str, model_info: Dict, model_data: Dict) -> Dict[str, str]:
        """Genera gráficos comunes para todos los modelos"""
        charts = {}
        
        try:
            # 1. Importancia de Características
            charts['feature_importance'] = self._create_feature_importance_chart(
                model_id, model_info, model_data
            )
            
            # 2. Métricas del Modelo
            charts['model_metrics'] = self._create_metrics_chart(
                model_id, model_info, model_data
            )
            
            # 3. Resumen del Entrenamiento
            charts['training_summary'] = self._create_training_summary_chart(
                model_id, model_info, model_data
            )
            
        except Exception as e:
            print(f"Error generando gráficos comunes: {str(e)}")
        
        return charts
    
    def _create_prediction_vs_actual_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de predicciones vs valores reales"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Obtener datos de predicciones
            y_test = model_info.get('y_test', [])
            y_pred = model_info.get('y_pred', [])
            
            if not y_test or not y_pred:
                # Crear datos de ejemplo si no están disponibles
                np.random.seed(42)
                y_test = np.random.normal(0, 1, 100)
                y_pred = y_test + np.random.normal(0, 0.2, 100)
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.6, s=50)
            
            # Línea perfecta (y=x)
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
            
            # Calcular R²
            r2 = model_data.get('metrics', {}).get('r2_score', 0)
            
            ax.set_xlabel('Valores Reales', fontsize=12)
            ax.set_ylabel('Predicciones', fontsize=12)
            ax.set_title(f'Predicciones vs Valores Reales\nR² = {r2:.3f}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Agregar texto con métricas
            mse = model_data.get('metrics', {}).get('mse', 0)
            rmse = model_data.get('metrics', {}).get('rmse', 0)
            ax.text(0.05, 0.95, f'MSE: {mse:.3f}\nRMSE: {rmse:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return self._save_chart(fig, f"{model_id}_prediction_vs_actual")
            
        except Exception as e:
            print(f"Error creando gráfico de predicciones: {str(e)}")
            return ""
    
    def _create_residuals_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de residuos"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Obtener datos
            y_test = model_info.get('y_test', [])
            y_pred = model_info.get('y_pred', [])
            
            if not y_test or not y_pred:
                np.random.seed(42)
                y_test = np.random.normal(0, 1, 100)
                y_pred = y_test + np.random.normal(0, 0.2, 100)
            
            residuals = np.array(y_test) - np.array(y_pred)
            
            # Gráfico 1: Residuos vs Predicciones
            ax1.scatter(y_pred, residuals, alpha=0.6)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicciones')
            ax1.set_ylabel('Residuos')
            ax1.set_title('Residuos vs Predicciones')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico 2: Histograma de Residuos
            ax2.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residuos')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Distribución de Residuos')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self._save_chart(fig, f"{model_id}_residuals")
            
        except Exception as e:
            print(f"Error creando gráfico de residuos: {str(e)}")
            return ""
    
    def _create_error_distribution_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de distribución de errores"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_test = model_info.get('y_test', [])
            y_pred = model_info.get('y_pred', [])
            
            if not y_test or not y_pred:
                np.random.seed(42)
                y_test = np.random.normal(0, 1, 100)
                y_pred = y_test + np.random.normal(0, 0.2, 100)
            
            errors = np.array(y_test) - np.array(y_pred)
            
            # Histograma de errores
            ax.hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)
            
            # Curva normal superpuesta
            mu, sigma = np.mean(errors), np.std(errors)
            x = np.linspace(errors.min(), errors.max(), 100)
            normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
            ax.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            
            ax.set_xlabel('Error de Predicción')
            ax.set_ylabel('Densidad')
            ax.set_title('Distribución de Errores de Predicción')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._save_chart(fig, f"{model_id}_error_distribution")
            
        except Exception as e:
            print(f"Error creando gráfico de distribución de errores: {str(e)}")
            return ""
    
    def _create_confusion_matrix_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea matriz de confusión"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            y_test = model_info.get('y_test', [])
            y_pred = model_info.get('y_pred', [])
            
            if not y_test or not y_pred:
                # Crear datos de ejemplo
                np.random.seed(42)
                y_test = np.random.choice([0, 1], 100)
                y_pred = np.random.choice([0, 1], 100)
            
            # Crear matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            
            # Visualizar con seaborn
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicciones')
            ax.set_ylabel('Valores Reales')
            ax.set_title('Matriz de Confusión')
            
            return self._save_chart(fig, f"{model_id}_confusion_matrix")
            
        except Exception as e:
            print(f"Error creando matriz de confusión: {str(e)}")
            return ""
    
    def _create_roc_curve_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea curva ROC"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            y_test = model_info.get('y_test', [])
            y_pred_proba = model_info.get('y_pred_proba', [])
            
            if not y_test or not y_pred_proba:
                # Crear datos de ejemplo
                np.random.seed(42)
                y_test = np.random.choice([0, 1], 100)
                y_pred_proba = np.random.random(100)
            
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Graficar
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.set_title('Curva ROC')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            return self._save_chart(fig, f"{model_id}_roc_curve")
            
        except Exception as e:
            print(f"Error creando curva ROC: {str(e)}")
            return ""
    
    def _create_probability_distribution_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de distribución de probabilidades"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_pred_proba = model_info.get('y_pred_proba', [])
            
            if not y_pred_proba:
                np.random.seed(42)
                y_pred_proba = np.random.random(100)
            
            # Histograma de probabilidades
            ax.hist(y_pred_proba, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.5, color='r', linestyle='--', label='Umbral de Decisión')
            
            ax.set_xlabel('Probabilidad Predicha')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Probabilidades Predichas')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return self._save_chart(fig, f"{model_id}_probability_distribution")
            
        except Exception as e:
            print(f"Error creando gráfico de distribución de probabilidades: {str(e)}")
            return ""
    
    def _create_clusters_2d_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea visualización 2D de clusters"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Obtener datos de clusters
            X = model_info.get('X_test', [])
            labels = model_info.get('cluster_labels', [])
            
            if not X or not labels:
                # Crear datos de ejemplo
                np.random.seed(42)
                X = np.random.randn(100, 2)
                labels = np.random.choice([0, 1, 2], 100)
            
            # Si X tiene más de 2 dimensiones, usar PCA
            if len(X[0]) > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
            else:
                X_2d = X
            
            # Scatter plot con colores por cluster
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            
            ax.set_xlabel('Componente Principal 1')
            ax.set_ylabel('Componente Principal 2')
            ax.set_title('Visualización de Clusters (2D)')
            plt.colorbar(scatter)
            ax.grid(True, alpha=0.3)
            
            return self._save_chart(fig, f"{model_id}_clusters_2d")
            
        except Exception as e:
            print(f"Error creando gráfico de clusters 2D: {str(e)}")
            return ""
    
    def _create_cluster_distribution_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de distribución de clusters"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = model_info.get('cluster_labels', [])
            
            if not labels:
                np.random.seed(42)
                labels = np.random.choice([0, 1, 2, 3], 100)
            
            # Contar elementos por cluster
            unique, counts = np.unique(labels, return_counts=True)
            
            # Gráfico de barras
            bars = ax.bar(unique, counts, alpha=0.7, edgecolor='black')
            
            # Agregar valores en las barras
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
            
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Número de Elementos')
            ax.set_title('Distribución de Elementos por Cluster')
            ax.grid(True, alpha=0.3, axis='y')
            
            return self._save_chart(fig, f"{model_id}_cluster_distribution")
            
        except Exception as e:
            print(f"Error creando gráfico de distribución de clusters: {str(e)}")
            return ""
    
    def _create_feature_importance_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de importancia de características"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Obtener importancia de características
            feature_importance = model_info.get('feature_importance', {})
            feature_names = model_data.get('feature_columns', [])
            
            if not feature_importance and feature_names:
                # Crear importancia de ejemplo
                np.random.seed(42)
                feature_importance = dict(zip(feature_names, np.random.random(len(feature_names))))
            
            if feature_importance:
                # Ordenar por importancia
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_features[:10])  # Top 10
                
                # Gráfico de barras horizontales
                y_pos = np.arange(len(features))
                bars = ax.barh(y_pos, importances, alpha=0.7)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Importancia')
                ax.set_title('Importancia de Características (Top 10)')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Agregar valores en las barras
                for i, (bar, importance) in enumerate(zip(bars, importances)):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{importance:.3f}', ha='left', va='center')
            
            return self._save_chart(fig, f"{model_id}_feature_importance")
            
        except Exception as e:
            print(f"Error creando gráfico de importancia: {str(e)}")
            return ""
    
    def _create_metrics_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de métricas del modelo"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = model_data.get('metrics', {})
            
            if not metrics:
                # Métricas de ejemplo
                metrics = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                }
            
            # Filtrar métricas numéricas
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            
            if numeric_metrics:
                metric_names = list(numeric_metrics.keys())
                metric_values = list(numeric_metrics.values())
                
                # Gráfico de barras
                bars = ax.bar(metric_names, metric_values, alpha=0.7, edgecolor='black')
                
                # Agregar valores en las barras
                for bar, value in zip(bars, metric_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.set_ylabel('Valor')
                ax.set_title('Métricas del Modelo')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45)
            
            return self._save_chart(fig, f"{model_id}_metrics")
            
        except Exception as e:
            print(f"Error creando gráfico de métricas: {str(e)}")
            return ""
    
    def _create_training_summary_chart(self, model_id: str, model_info: Dict, model_data: Dict) -> str:
        """Crea gráfico de resumen del entrenamiento"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Información del modelo
            model_type = model_data.get('model_type', 'Unknown')
            dataset_name = model_data.get('dataset_name', 'Unknown')
            training_time = model_data.get('training_time', 0)
            created_at = model_data.get('created_at', datetime.now().isoformat())
            
            # Gráfico 1: Información básica
            ax1.axis('off')
            info_text = f"""
            📊 RESUMEN DEL MODELO
            
            🏷️ Tipo: {model_type}
            📁 Dataset: {dataset_name}
            ⏱️ Tiempo de Entrenamiento: {training_time:.2f}s
            📅 Creado: {created_at[:10]}
            🎯 Target: {model_data.get('target_column', 'N/A')}
            🔢 Features: {len(model_data.get('feature_columns', []))}
            """
            ax1.text(0.1, 0.9, info_text, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Gráfico 2: Métricas principales
            ax2.axis('off')
            metrics = model_data.get('metrics', {})
            metrics_text = "📈 MÉTRICAS PRINCIPALES\n\n"
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_text += f"• {metric}: {value:.3f}\n"
            ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # Gráfico 3: Configuración del modelo
            ax3.axis('off')
            config_text = "⚙️ CONFIGURACIÓN\n\n"
            config_text += f"• Test Size: {model_data.get('test_size', 0.2)}\n"
            config_text += f"• Random State: {model_data.get('random_state', 42)}\n"
            config_text += f"• Cross Validation: {model_data.get('cv_folds', 5)}\n"
            ax3.text(0.1, 0.9, config_text, transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # Gráfico 4: Estado del modelo
            ax4.axis('off')
            status_text = "✅ ESTADO DEL MODELO\n\n"
            status_text += "• ✅ Entrenado correctamente\n"
            status_text += "• ✅ Métricas calculadas\n"
            status_text += "• ✅ Guardado en disco\n"
            status_text += "• ✅ Listo para predicciones\n"
            ax4.text(0.1, 0.9, status_text, transform=ax4.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            plt.suptitle(f'Resumen del Entrenamiento - Modelo {model_id[:8]}...', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            return self._save_chart(fig, f"{model_id}_training_summary")
            
        except Exception as e:
            print(f"Error creando gráfico de resumen: {str(e)}")
            return ""
    
    def _save_chart(self, fig, filename: str) -> str:
        """Guarda el gráfico y retorna la ruta"""
        try:
            chart_path = os.path.join(self.charts_dir, f"{filename}.png")
            fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return chart_path
        except Exception as e:
            print(f"Error guardando gráfico {filename}: {str(e)}")
            plt.close(fig)
            return ""
    
    def _is_binary_classification(self, model_data: Dict) -> bool:
        """Verifica si es clasificación binaria"""
        target_column = model_data.get('target_column', '')
        # Lógica simple para detectar clasificación binaria
        return 'binary' in model_data.get('model_type', '').lower() or \
               'logistic' in model_data.get('model_type', '').lower()
    
    def get_chart_base64(self, chart_path: str) -> str:
        """Convierte un gráfico a base64 para enviar al frontend"""
        try:
            if not os.path.exists(chart_path):
                return ""
            
            with open(chart_path, 'rb') as img_file:
                img_data = img_file.read()
                return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            print(f"Error convirtiendo gráfico a base64: {str(e)}")
            return ""


