"""
Servicio de Agente Inteligente para guiar al usuario
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AgentService:
    """Agente inteligente que guía al usuario en el proceso de ML"""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
    
    def analyze_dataset(self, file_path: str) -> Dict[str, Any]:
        """Analiza un dataset y proporciona sugerencias"""
        try:
            # Cargar dataset
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return {
                    "success": False,
                    "message": "Formato de archivo no soportado"
                }
            
            analysis = {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicates": int(df.duplicated().sum())
            }
            
            # Generar sugerencias basadas en el análisis
            suggestions = self._generate_suggestions(analysis)
            next_step = self._determine_next_step(analysis)
            warnings = self._generate_warnings(analysis)
            
            return {
                "success": True,
                "analysis": analysis,
                "suggestions": suggestions,
                "next_step": next_step,
                "warnings": warnings
            }
        
        except Exception as e:
            logger.error(f"Error al analizar dataset: {str(e)}")
            return {
                "success": False,
                "message": f"Error al analizar dataset: {str(e)}"
            }
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Genera sugerencias basadas en el análisis"""
        suggestions = []
        
        # Duplicados
        if analysis["duplicates"] > 0:
            suggestions.append(
                f"📊 Se detectaron {analysis['duplicates']} filas duplicadas. "
                f"Te recomiendo eliminarlas para mejorar la calidad de los datos."
            )
        
        # Valores faltantes
        missing_cols = [col for col, count in analysis["missing_values"].items() if count > 0]
        if missing_cols:
            total_missing = sum(analysis["missing_values"].values())
            suggestions.append(
                f"⚠️ Hay {total_missing} valores faltantes en {len(missing_cols)} columnas. "
                f"Puedes rellenarlos con la media/mediana o eliminar las filas."
            )
        
        # Columnas categóricas
        if len(analysis["categorical_columns"]) > 0:
            suggestions.append(
                f"🔤 Hay {len(analysis['categorical_columns'])} columnas categóricas. "
                f"Necesitarás codificarlas (Label Encoding o One-Hot) antes de entrenar."
            )
        
        # Dataset pequeño
        if analysis["rows"] < 100:
            suggestions.append(
                f"⚠️ El dataset es pequeño ({analysis['rows']} filas). "
                f"Esto puede limitar la precisión de los modelos. Considera usar validación cruzada."
            )
        
        # Dataset grande
        if analysis["rows"] > 100000:
            suggestions.append(
                f"💪 Tienes un dataset grande ({analysis['rows']} filas). "
                f"Podrías beneficiarte de modelos más complejos como Gradient Boosting o Redes Neuronales."
            )
        
        # Muchas columnas
        if analysis["columns"] > 50:
            suggestions.append(
                f"📈 Tienes {analysis['columns']} columnas. "
                f"Considera aplicar selección de características o reducción de dimensionalidad (PCA)."
            )
        
        return suggestions
    
    def _determine_next_step(self, analysis: Dict[str, Any]) -> str:
        """Determina el siguiente paso recomendado"""
        
        # Si hay duplicados, ese es el primer paso
        if analysis["duplicates"] > 0:
            return "clean_duplicates"
        
        # Si hay valores faltantes
        missing_count = sum(analysis["missing_values"].values())
        if missing_count > 0:
            return "handle_missing_values"
        
        # Si hay columnas categóricas sin codificar
        if len(analysis["categorical_columns"]) > 0:
            return "encode_categorical"
        
        # Si ya está limpio, normalizar
        if len(analysis["numeric_columns"]) > 0:
            return "normalize_data"
        
        # Listo para entrenar
        return "ready_to_train"
    
    def _generate_warnings(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Genera advertencias importantes"""
        warnings = []
        
        # Dataset muy pequeño
        if analysis["rows"] < 30:
            warnings.append(
                "⚠️ ADVERTENCIA: El dataset es muy pequeño para entrenar modelos de ML confiables."
            )
        
        # Demasiados valores faltantes
        total_values = analysis["rows"] * analysis["columns"]
        total_missing = sum(analysis["missing_values"].values())
        missing_percentage = (total_missing / total_values) * 100
        
        if missing_percentage > 30:
            warnings.append(
                f"⚠️ ADVERTENCIA: {missing_percentage:.1f}% de los datos están faltantes. "
                f"Esto puede afectar seriamente la calidad del modelo."
            )
        
        # Sin columnas numéricas
        if len(analysis["numeric_columns"]) == 0:
            warnings.append(
                "⚠️ ADVERTENCIA: No hay columnas numéricas. "
                "Necesitas al menos algunas variables numéricas para entrenar la mayoría de modelos."
            )
        
        return " | ".join(warnings) if warnings else None
    
    def get_cleaning_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Proporciona recomendaciones específicas para limpieza"""
        recommendations = {
            "priority_actions": [],
            "optional_actions": [],
            "reasons": {}
        }
        
        # Acciones prioritarias
        if analysis["duplicates"] > 0:
            recommendations["priority_actions"].append("remove_duplicates")
            recommendations["reasons"]["remove_duplicates"] = \
                "Eliminar duplicados mejora la calidad y evita sesgo en el entrenamiento"
        
        missing_count = sum(analysis["missing_values"].values())
        if missing_count > 0:
            # Determinar mejor estrategia
            if missing_count < analysis["rows"] * 0.05:
                recommendations["priority_actions"].append("drop_nulls")
                recommendations["reasons"]["drop_nulls"] = \
                    "Pocos valores faltantes (<5%), es seguro eliminar esas filas"
            else:
                recommendations["priority_actions"].append("fill_nulls_mean")
                recommendations["reasons"]["fill_nulls_mean"] = \
                    "Muchos valores faltantes, mejor rellenar con media/mediana"
        
        if len(analysis["categorical_columns"]) > 0:
            recommendations["priority_actions"].append("encode_categorical")
            recommendations["reasons"]["encode_categorical"] = \
                "Los modelos necesitan datos numéricos"
        
        # Acciones opcionales
        if len(analysis["numeric_columns"]) > 0:
            recommendations["optional_actions"].append("normalize")
            recommendations["reasons"]["normalize"] = \
                "La normalización ayuda a modelos como SVM y Redes Neuronales"
            
            recommendations["optional_actions"].append("remove_outliers")
            recommendations["reasons"]["remove_outliers"] = \
                "Eliminar outliers puede mejorar la robustez del modelo"
        
        return recommendations
    
    def suggest_model_type(self, target_column: str, df: pd.DataFrame = None, 
                          file_path: str = None) -> Dict[str, Any]:
        """Sugiere el tipo de modelo basado en la columna objetivo"""
        try:
            if df is None and file_path:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
            
            if df is None or target_column not in df.columns:
                return {
                    "success": False,
                    "message": "No se pudo analizar la columna objetivo"
                }
            
            target = df[target_column]
            unique_values = target.nunique()
            total_values = len(target)
            
            suggestions = {
                "success": True,
                "target_column": target_column,
                "unique_values": int(unique_values),
                "recommended_type": None,
                "recommended_algorithms": [],
                "explanation": ""
            }
            
            # Determinar tipo de problema
            if target.dtype in [np.float64, np.int64]:
                if unique_values > 20:
                    # Regresión
                    suggestions["recommended_type"] = "regression"
                    suggestions["recommended_algorithms"] = [
                        "linear_regression",
                        "random_forest_regressor",
                        "gradient_boosting_regressor",
                        "neural_network"
                    ]
                    suggestions["explanation"] = \
                        f"La columna '{target_column}' es numérica con {unique_values} valores únicos. " \
                        f"Esto sugiere un problema de REGRESIÓN."
                else:
                    # Clasificación
                    suggestions["recommended_type"] = "classification"
                    suggestions["recommended_algorithms"] = [
                        "logistic_regression",
                        "random_forest_classifier",
                        "gradient_boosting_classifier"
                    ]
                    suggestions["explanation"] = \
                        f"La columna '{target_column}' tiene {unique_values} valores únicos. " \
                        f"Esto sugiere un problema de CLASIFICACIÓN."
            else:
                # Categórico = Clasificación
                suggestions["recommended_type"] = "classification"
                
                if unique_values == 2:
                    suggestions["recommended_algorithms"] = [
                        "logistic_regression",
                        "svc",
                        "random_forest_classifier"
                    ]
                    suggestions["explanation"] = \
                        f"La columna '{target_column}' es binaria (2 clases). " \
                        f"Clasificación binaria."
                else:
                    suggestions["recommended_algorithms"] = [
                        "random_forest_classifier",
                        "gradient_boosting_classifier",
                        "neural_network"
                    ]
                    suggestions["explanation"] = \
                        f"La columna '{target_column}' tiene {unique_values} clases. " \
                        f"Clasificación multiclase."
            
            return suggestions
        
        except Exception as e:
            logger.error(f"Error al sugerir tipo de modelo: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }
    
    def process_user_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Procesa un mensaje del usuario y proporciona una respuesta inteligente"""
        
        message_lower = message.lower()
        
        # Agregar a historial
        self.conversation_history.append({
            "role": "user",
            "message": message
        })
        
        response = {
            "success": True,
            "message": "",
            "suggestions": [],
            "next_step": None
        }
        
        # Saludos
        if any(word in message_lower for word in ["hola", "hello", "hi", "buenas", "saludos"]):
            response["message"] = \
                "¡Hola! 👋 ¡Qué gusto verte por aquí!\n\n" \
                "Soy tu **Agente Inteligente de Machine Learning**. Estoy aquí para guiarte en cada paso del proceso.\n\n" \
                "✨ Puedo ayudarte a transformar tus datos en modelos predictivos poderosos.\n\n" \
                "¿Quieres saber cómo funciona el sistema o prefieres empezar directamente?"
            
            response["suggestions"] = [
                "¿Cómo funciona el sistema?",
                "Quiero empezar ahora",
                "¿Qué puedo hacer aquí?",
                "Necesito ayuda"
            ]
        
        # Sistema y funcionamiento
        elif any(word in message_lower for word in ["funciona", "sistema", "plataforma", "cómo funciona", "como funciona", "qué es", "que es"]):
            response["message"] = \
                "🚀 **Así funciona nuestro sistema de ML:**\n\n" \
                "**PASO 1: Cargar Datos 📤**\n" \
                "Sube tu archivo CSV o Excel con tus datos. Acepto prácticamente cualquier formato.\n\n" \
                "**PASO 2: Limpiar Datos 🧹**\n" \
                "Yo analizo tus datos y te sugiero qué limpiar:\n" \
                "• Elimino duplicados\n" \
                "• Manejo valores faltantes\n" \
                "• Codifico variables categóricas\n" \
                "• Normalizo números\n\n" \
                "**PASO 3: Entrenar Modelo 🤖**\n" \
                "Tú eliges:\n" \
                "• La variable a predecir\n" \
                "• Las características a usar\n" \
                "• El algoritmo (te recomiendo el mejor)\n\n" \
                "**PASO 4: Ver Resultados 📊**\n" \
                "Te muestro métricas, predicciones y gráficos para que entiendas tu modelo.\n\n" \
                "💡 **Lo mejor:** Todo con librerías profesionales (pandas, scikit-learn, PyTorch).\n\n" \
                "¿Por dónde quieres empezar?"
            
            response["suggestions"] = [
                "Quiero subir datos",
                "¿Qué algoritmos tienes?",
                "¿Cómo limpio datos?",
                "Ver ejemplo completo"
            ]
        
        # Ayuda general
        elif any(word in message_lower for word in ["ayuda", "help", "qué hacer", "que hacer", "comenzar", "empezar", "puedo hacer"]):
            response["message"] = \
                "🎯 **Aquí está todo lo que puedes hacer:**\n\n" \
                "**1️⃣ DASHBOARD**\n" \
                "Ve estadísticas en tiempo real de tus proyectos y actividad.\n\n" \
                "**2️⃣ CARGAR DATOS**\n" \
                "• Sube CSV o Excel\n" \
                "• Yo analizo automáticamente\n" \
                "• Te doy un resumen completo\n\n" \
                "**3️⃣ LIMPIAR DATOS**\n" \
                "• Eliminar duplicados\n" \
                "• Rellenar valores nulos\n" \
                "• Normalizar y estandarizar\n" \
                "• Codificar variables categóricas\n" \
                "• Remover outliers\n\n" \
                "**4️⃣ ENTRENAR MODELOS**\n" \
                "• Regresión (predecir números)\n" \
                "• Clasificación (predecir categorías)\n" \
                "• Clustering (agrupar datos)\n" \
                "• Redes Neuronales (PyTorch)\n\n" \
                "**5️⃣ CONFIGURACIÓN**\n" \
                "Ajusta parámetros del sistema.\n\n" \
                "¿Te gustaría profundizar en alguno?"
            
            response["suggestions"] = [
                "Explícame la limpieza",
                "¿Qué algoritmos hay?",
                "¿Cómo funciona el sistema?",
                "Dame un ejemplo"
            ]
        
        # Limpieza de datos
        elif any(word in message_lower for word in ["limpiar", "clean", "preprocesar", "limpieza"]):
            response["message"] = \
                "🧹 **Guía de Limpieza de Datos:**\n\n" \
                "**1. Eliminar Duplicados**\n" \
                "Detecta y elimina filas repetidas que pueden sesgar tu modelo.\n\n" \
                "**2. Valores Faltantes**\n" \
                "Opciones:\n" \
                "• Rellenar con media/mediana (para números)\n" \
                "• Rellenar con moda (para categorías)\n" \
                "• Eliminar filas (si son pocas)\n\n" \
                "**3. Codificar Categóricas**\n" \
                "Convierte texto a números:\n" \
                "• 'Hombre' → 0, 'Mujer' → 1\n" \
                "Los modelos solo entienden números 🔢\n\n" \
                "**4. Normalizar (0-1)**\n" \
                "Escala valores: edad (18-80) → (0.0-1.0)\n" \
                "Útil para SVM y Redes Neuronales.\n\n" \
                "**5. Estandarizar (Z-score)**\n" \
                "Centra datos en media=0, desviación=1\n\n" \
                "**6. Eliminar Outliers**\n" \
                "Remueve valores extremos que distorsionan.\n\n" \
                "💡 **Mi recomendación:** Déjame analizar tu dataset y te digo qué necesitas limpiar."
            
            response["next_step"] = "clean"
            response["suggestions"] = [
                "¿Qué son outliers?",
                "¿Cuándo normalizar?",
                "Ya tengo datos, ¿qué sigue?",
                "Ver ejemplo de limpieza"
            ]
        
        # Entrenamiento
        elif any(word in message_lower for word in ["entrenar", "train", "modelo", "model", "algoritmo"]):
            response["message"] = \
                "🤖 **Entrenar Modelos - Guía Completa:**\n\n" \
                "**¿Qué tipo de problema tienes?**\n\n" \
                "📊 **REGRESIÓN** (predecir números)\n" \
                "Ej: precio de casa, ventas, temperatura\n" \
                "Algoritmos:\n" \
                "• Linear Regression (simple y rápido)\n" \
                "• Random Forest (preciso y robusto)\n" \
                "• Gradient Boosting (muy preciso)\n" \
                "• Redes Neuronales (datos complejos)\n\n" \
                "🎯 **CLASIFICACIÓN** (predecir categorías)\n" \
                "Ej: spam/no spam, enfermo/sano, aprobar/reprobar\n" \
                "Algoritmos:\n" \
                "• Logistic Regression (rápido)\n" \
                "• Random Forest (versátil)\n" \
                "• SVM (preciso con datos limpios)\n" \
                "• Redes Neuronales (imágenes, texto)\n\n" \
                "🔮 **CLUSTERING** (agrupar similares)\n" \
                "Ej: segmentar clientes, detectar patrones\n" \
                "Algoritmos:\n" \
                "• K-Means (grupos definidos)\n" \
                "• DBSCAN (formas irregulares)\n\n" \
                "💡 **Tip:** Dime tu columna objetivo y te recomiendo el mejor modelo."
            
            response["next_step"] = "train"
            response["suggestions"] = [
                "¿Cuál algoritmo es mejor?",
                "¿Qué es Random Forest?",
                "¿Regresión o clasificación?",
                "Ya tengo datos limpios"
            ]
        
        # Métricas
        elif any(word in message_lower for word in ["metrics", "métricas", "resultados", "accuracy", "precision", "mse", "r2"]):
            response["message"] = \
                "📊 **Entendiendo las Métricas:**\n\n" \
                "**REGRESIÓN:**\n\n" \
                "🔹 **MSE** (Mean Squared Error)\n" \
                "Error cuadrático. Menor = mejor.\n" \
                "Penaliza errores grandes.\n\n" \
                "🔹 **RMSE** (Root MSE)\n" \
                "√MSE. Está en las mismas unidades que tu variable.\n\n" \
                "🔹 **MAE** (Mean Absolute Error)\n" \
                "Error promedio absoluto. Más fácil de interpretar.\n\n" \
                "🔹 **R²** (R-cuadrado)\n" \
                "0-1 (0-100%). Cuánto explica tu modelo.\n" \
                "• 0.9 = 90% explicado (excelente)\n" \
                "• 0.5 = 50% explicado (regular)\n\n" \
                "**CLASIFICACIÓN:**\n\n" \
                "🔹 **Accuracy**\n" \
                "% de predicciones correctas.\n" \
                "0.85 = 85% correcto.\n\n" \
                "🔹 **Precision**\n" \
                "De los que predije positivos, cuántos eran realmente positivos.\n\n" \
                "🔹 **Recall**\n" \
                "De todos los positivos reales, cuántos detecté.\n\n" \
                "🔹 **F1-Score**\n" \
                "Balance entre Precision y Recall.\n\n" \
                "💡 **¿Qué es bueno?** Depende del problema, pero en general:\n" \
                "• Accuracy > 0.80 → Bueno\n" \
                "• Accuracy > 0.90 → Excelente\n" \
                "• Accuracy > 0.95 → Excepcional"
            
            response["suggestions"] = [
                "¿Cuál es más importante?",
                "¿Qué significa R²?",
                "Mi accuracy es 0.75, ¿es bueno?",
                "Ver ejemplo"
            ]
        
        # Algoritmos específicos
        elif any(word in message_lower for word in ["random forest", "gradient boosting", "svm", "neural", "red neuronal"]):
            response["message"] = \
                "🧠 **Algoritmos Populares:**\n\n" \
                "**🌲 Random Forest**\n" \
                "• Crea muchos árboles de decisión\n" \
                "• Vota entre todos\n" \
                "• Muy robusto, difícil de sobreajustar\n" \
                "• Bueno para: Casi todo\n\n" \
                "**⚡ Gradient Boosting**\n" \
                "• Árboles que aprenden de errores anteriores\n" \
                "• Muy preciso\n" \
                "• Más lento que Random Forest\n" \
                "• Bueno para: Competencias, alta precisión\n\n" \
                "**🎯 SVM (Support Vector Machine)**\n" \
                "• Busca el mejor hiperplano separador\n" \
                "• Excelente con datos bien separados\n" \
                "• Requiere normalización\n" \
                "• Bueno para: Clasificación binaria\n\n" \
                "**🧠 Redes Neuronales (PyTorch)**\n" \
                "• Inspiradas en el cerebro humano\n" \
                "• Aprenden patrones complejos\n" \
                "• Necesitan muchos datos\n" \
                "• Bueno para: Imágenes, texto, patrones complejos\n\n" \
                "💡 **¿Cuál usar?** Yo te recomiendo según tus datos."
            
            response["suggestions"] = [
                "¿Cuál es el más preciso?",
                "¿Cuál es el más rápido?",
                "Tengo 1000 filas, ¿cuál uso?",
                "Quiero entrenar ahora"
            ]
        
        # Subir datos
        elif any(word in message_lower for word in ["subir", "upload", "cargar", "archivo", "csv", "excel"]):
            response["message"] = \
                "📤 **Cómo Subir tus Datos:**\n\n" \
                "**Formatos Aceptados:**\n" \
                "✅ CSV (.csv) - Recomendado\n" \
                "✅ Excel (.xlsx, .xls)\n\n" \
                "**Pasos:**\n" \
                "1. Ve a 'Cargar Datos' en el menú\n" \
                "2. Click en el área de carga o arrastra tu archivo\n" \
                "3. Yo automáticamente:\n" \
                "   • Leo tu archivo (incluso con punto y coma ';')\n" \
                "   • Detecto tipos de columnas\n" \
                "   • Cuento valores faltantes\n" \
                "   • Te doy recomendaciones\n\n" \
                "**Tips:**\n" \
                "• Primera fila debe tener nombres de columnas\n" \
                "• Evita caracteres especiales en nombres\n" \
                "• No hay límite de filas (recomiendo < 1M)\n\n" \
                "¿Tienes tu archivo listo?"
            
            response["suggestions"] = [
                "¿Y si tengo punto y coma?",
                "¿Cuántas filas soporta?",
                "Ya subí, ¿qué sigue?",
                "Ver ejemplo de archivo"
            ]
        
        # Agradecimientos
        elif any(word in message_lower for word in ["gracias", "thanks", "thank you", "genial", "excelente", "perfecto"]):
            response["message"] = \
                "¡De nada! 😊 ¡Estoy aquí para ayudarte!\n\n" \
                "¿Hay algo más que quieras saber? Puedo ayudarte con:\n" \
                "• Dudas técnicas\n" \
                "• Recomendaciones personalizadas\n" \
                "• Solución de problemas\n" \
                "• Lo que necesites\n\n" \
                "¡Pregunta con confianza! 🚀"
            
            response["suggestions"] = [
                "¿Cómo mejoro mi modelo?",
                "¿Qué sigue después de entrenar?",
                "Ver dashboard",
                "Listo, voy a probar"
            ]
        
        else:
            # Respuesta inteligente por defecto
            response["message"] = \
                "🤔 Hmm, no estoy seguro de entender completamente.\n\n" \
                "Pero déjame ayudarte. ¿Tu pregunta es sobre:\n\n" \
                "📤 **Subir datos** - Cómo cargar archivos\n" \
                "🧹 **Limpiar datos** - Preprocesamiento\n" \
                "🤖 **Entrenar modelos** - Algoritmos y ML\n" \
                "📊 **Métricas** - Interpretar resultados\n" \
                "❓ **Otra cosa** - Dime más detalles\n\n" \
                "Puedes preguntarme de forma natural, como:\n" \
                "• '¿Cómo funciona el sistema?'\n" \
                "• '¿Qué algoritmo debo usar?'\n" \
                "• 'Explícame Random Forest'\n" \
                "• 'Tengo 1000 filas, ¿qué hago?'"
            
            response["suggestions"] = [
                "¿Cómo funciona el sistema?",
                "¿Cómo limpio mis datos?",
                "¿Qué modelo usar?",
                "Ayuda general"
            ]
        
        # Agregar respuesta al historial
        self.conversation_history.append({
            "role": "assistant",
            "message": response["message"]
        })
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Obtiene el historial de conversación"""
        return self.conversation_history
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
        logger.info("Historial de conversación limpiado")

